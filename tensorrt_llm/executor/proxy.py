import atexit
import concurrent.futures
import threading
import time
import traceback
import weakref
from typing import Dict, Optional, Union
import asyncio
import json
from pathlib import Path

import torch
import zmq
import zmq.asyncio

from tensorrt_llm.logger import logger

from .._utils import customized_gc_thresholds, mpi_rank, nvtx_range_debug
from ..llmapi.mpi_session import (MpiCommSession, MpiPoolSession, MpiSession,
                                  RemoteMpiCommSessionClient)
from ..llmapi.tracer import enable_llm_tracer, get_tracer, global_tracer
from ..llmapi.utils import (AsyncQueue, ManagedThread, _SyncQueue,
                            enable_llm_debug, print_colored,
                            print_colored_debug)
from .executor import GenerationExecutor
from .ipc import FusedIpcQueue, IpcQueue
from .postproc_worker import PostprocWorkerConfig
from .request import (CancellingRequest, GenerationRequest, LoadStatsRequest,
                      LoadModelRequest, UnloadModelRequest)
from .result import GenerationResult, IterationResult
from .utils import (ErrorResponse, IntraProcessQueue, WorkerCommIpcAddrs,
                    create_mpi_comm_session, get_spawn_proxy_process_env,
                    is_llm_response, print_alive_threads)
from .worker import GenerationExecutorWorker, worker_main

__all__ = [
    "GenerationExecutorProxy",
]


class GenerationExecutorProxy(GenerationExecutor):
    READY_SIGNAL = b"READY"
    PAUSED_SIGNAL = b"PAUSED_NO_MODEL"

    def __init__(
        self,
        worker_kwargs: dict,
        model_world_size: int = 1,
        mpi_session: Optional[MpiSession] = None,
        *,
        worker_cls: type = GenerationExecutorWorker,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
        garbage_collection_gen0_threshold: Optional[int] = None,
        lazy_load: bool = False,
        max_num_workers: Optional[int] = None,
    ) -> None:
        # Check parameters: max_num_workers should be provided only when lazy_load is True
        if lazy_load:
            assert max_num_workers is not None, "max_num_workers must be provided when lazy_load is True."
        else:
            assert max_num_workers is None, "max_num_workers should not be provided when lazy_load is False."

        postproc_worker_config = postproc_worker_config or PostprocWorkerConfig(
        )
        super().__init__(
            num_postprocess_workers=postproc_worker_config.
            num_postprocess_workers,
            postprocess_tokenizer_dir=postproc_worker_config.
            postprocess_tokenizer_dir,
            is_llm_executor=is_llm_executor,
        )

        # Proxy 自身状态字段，防止重复启动 Worker
        self.workers_started = False

        # type = GenerationExecutorWorker
        self.worker_cls = worker_cls

        self.max_num_workers = max_num_workers

        mpi_process_pre_spawned: bool = get_spawn_proxy_process_env()

        if mpi_session is None:
            if mpi_process_pre_spawned:
                print_colored_debug('create comm session ...\n', "yellow")
                self.mpi_session = create_mpi_comm_session(model_world_size)
            else:
                print_colored_debug('create pool session ...\n', "yellow")
                if lazy_load:
                    print_colored_debug(
                        f'Using MpiPoolSession with max_num_workers={max_num_workers} for lazy loading...\n',
                        "yellow")
                    self.mpi_session = MpiPoolSession(n_workers=max_num_workers)
                else:
                    self.mpi_session = MpiPoolSession(n_workers=model_world_size)
        else:
            print_colored_debug('using external mpi session ...\n', "yellow")
            self.mpi_session = mpi_session

        if isinstance(self.mpi_session,
                      (MpiCommSession, RemoteMpiCommSessionClient)):
            print_colored(
                f"rank {mpi_rank()} using MpiCommSession to bind to external MPI processes\n",
                "yellow")
        else:
            print_colored(
                f"rank {mpi_rank()} using MpiPoolSession to spawn MPI processes\n",
                "yellow")

        self._results: Dict[int, GenerationResult] = {}

        if lazy_load:
            self.model_world_size = None  # unknown at init time
        else:
            self.model_world_size = model_world_size

        self.garbage_collection_gen0_threshold = garbage_collection_gen0_threshold

        self.lazy_load = lazy_load

        worker_kwargs = dict(**worker_kwargs,
                             worker_queues=self._setup_queues(),
                             postproc_worker_config=postproc_worker_config,
                             is_llm_executor=False,
                             garbage_collection_gen0_threshold=self.
                             garbage_collection_gen0_threshold,
                             lazy_load=lazy_load,)

        if "log_level" not in worker_kwargs:
            worker_kwargs["log_level"] = logger.level

        self.dispatch_result_thread: Optional[ManagedThread] = None
        self.dispatch_stats_thread: Optional[ManagedThread] = None
        self.dispatch_kv_cache_events_thread: Optional[ManagedThread] = None
        self._start_executor_workers(worker_kwargs)

        # MPI registers its joiner using threading._register_atexit if possible.
        # These functions run before atexit.register, so to avoid deadlock,
        # we have to notify workers to exit before MPI starts to wait them.
        try:
            threading._register_atexit(  # type: ignore[attr-defined]
                self.pre_shutdown)
        except AttributeError:
            atexit.register(self.pre_shutdown)

    def _setup_queues(self) -> WorkerCommIpcAddrs:
        self.request_queue = IpcQueue(is_server=True,
                                      name="proxy_request_queue")
        self.worker_init_status_queue = IpcQueue(
            is_server=True, name="worker_init_status_queue")
        # TODO[chunweiy]: Unify IpcQueue and FusedIpcQueue
        # Use PULL mode when enable_postprocess_parallel as there are
        # multiple senders from multiple processes.
        self.result_queue = FusedIpcQueue(
            is_server=True,
            fuse_message=False,
            socket_type=zmq.PULL
            if self.enable_postprocess_parallel else zmq.PAIR,
            name="proxy_result_queue"
        )
        self.mp_stats_queue = FusedIpcQueue(
            is_server=True,
            fuse_message=False,
            name="proxy_stats_queue")
        self.load_stats_queue = FusedIpcQueue(
            is_server=True,
            fuse_message=False,
            name="proxy_load_stats_queue")
        self.kv_cache_events_queue = FusedIpcQueue(
            is_server=True,
            fuse_message=False,
            name="proxy_kv_cache_events_queue")

        # 新增 control queue，用于 proxy -> worker 控制指令（例如 LOAD_MODEL）
        # 使用 IpcQueue server 端，worker 端以 client 方式连接（is_server=False）
        if self.lazy_load:
            self.control_queue_lst = []
            for i in range(self.max_num_workers):
                control_queue = IpcQueue(
                    is_server=True,
                    name=f"proxy_control_queue_{i}")
                self.control_queue_lst.append(control_queue)

        return WorkerCommIpcAddrs(
            request_queue_addr=self.request_queue.address,
            worker_init_status_queue_addr=self.worker_init_status_queue.address,
            result_queue_addr=self.result_queue.address,
            stats_queue_addr=self.mp_stats_queue.address,
            load_stats_queue_addr=self.load_stats_queue.address,
            kv_cache_events_queue_addr=self.kv_cache_events_queue.address,
            control_queue_addrs=[control_queue.address for control_queue in self.control_queue_lst] \
                                 if self.lazy_load else None,
        )

    def abort_request(self, request_id: int) -> None:
        ''' Abort a request by sending a cancelling request to the request queue.

        Args:
            request_id (int): The id of the request to abort.
        '''
        # NOTE, it just sends a cancelling request to the request queue, but it
        # may take a while for the request to be cancelled in the worker and
        # send back a finished result.
        self.request_queue.put(CancellingRequest(request_id))

    def dispatch_result_task(self) -> bool:
        # TODO[chunweiy]: convert the dispatch_result_task to async, that should
        # benefit from zmq.asyncio.Context
        with customized_gc_thresholds(self.garbage_collection_gen0_threshold):
            if (res := self.result_queue.get()) is None:
                return False  # shutdown the thread

        async_queues = []
        event_loop = None

        def process_res(res):
            client_id = res.client_id
            nonlocal event_loop
            nonlocal async_queues

            queue = self._results[client_id].queue
            if isinstance(queue, _SyncQueue):
                queue.put_nowait(res)
                async_queues.append(queue)
                # all the loops are identical
                event_loop = event_loop or queue.loop
            else:
                queue.put(res)

            if (is_llm_response(res) and res.result.is_final) or isinstance(
                    res, ErrorResponse):
                self._results.pop(client_id)

        res = res if isinstance(res, list) else [res]

        for i in res:
            global_tracer().log_instant("IPC.get")
            if i is None:
                return False
            process_res(i)

        if async_queues:
            _SyncQueue.notify_many(event_loop, async_queues)

        return True  # success

    def _iteration_result_task(self, queue: Union[FusedIpcQueue,
                                                  IntraProcessQueue],
                               result_singleton: IterationResult) -> bool:
        # iteration result is not urgent, so we can sleep a bit
        time.sleep(0.2)

        try:
            data = queue.get()
        except:
            logger.debug(
                "proxy.py: Error in _iteration_result_task: queue.get()")
            return False

        if data is None:
            logger.debug("proxy.py: _iteration_result_task: data is None")
            return False  # shutdown the thread

        data = data if isinstance(data, list) else [data]
        queue = result_singleton.queue
        async_queues = []

        while queue.full():
            queue.get()

        try:
            for d in data:
                if d is None:
                    logger.debug("proxy.py: _iteration_result_task: d is None")
                    return False

                if isinstance(queue, _SyncQueue):
                    queue.put_nowait(d)
                    async_queues.append(queue)
                else:
                    queue.put(d)

            if async_queues:
                _SyncQueue.notify_many(queue.loop, async_queues)

        except AsyncQueue.EventLoopShutdownError:
            # This happens in the last loop while the generate workflow is
            # stopped, or when get_stats() or aget_stats() are not called by users
            # and therefore event loop can already be closed.
            logger.debug("proxy.py: EventLoopShutdownError")
        except Exception as e:
            logger.debug(f"proxy.py: Error in _iteration_result_task: {e}")
            raise e

        return True  # success

    def dispatch_stats_task(self) -> bool:
        if not self._iter_stats_result:
            # This can happen temporarily because the WAR in tensorrt_llm/bench/benchmark/throughput.py
            # is not synchronized with self.dispatch_stats_thread.
            logger.debug(
                f"Skipping stats dispatch while self._iter_stats_result=None")
            return True  # Intended behavior, not an error
        return self._iteration_result_task(self.mp_stats_queue,
                                           self._iter_stats_result)

    def dispatch_kv_cache_events_task(self) -> bool:
        return self._iteration_result_task(self.kv_cache_events_queue,
                                           self._iter_kv_events_result)

    def _start_dispatch_threads(self):
        if self.dispatch_result_thread is None:

            self.dispatch_result_thread = ManagedThread(
                weakref.WeakMethod(self.dispatch_result_task),
                error_queue=self._error_queue,
                name="proxy_dispatch_result_thread")
            self.dispatch_stats_thread = ManagedThread(
                weakref.WeakMethod(self.dispatch_stats_task),
                error_queue=self._error_queue,
                name="proxy_dispatch_stats_thread")
            self.dispatch_kv_cache_events_thread = ManagedThread(
                weakref.WeakMethod(self.dispatch_kv_cache_events_task),
                error_queue=self._error_queue,
                name="proxy_dispatch_kv_cache_events_thread")

            self.dispatch_result_thread.start()

            # Only collect stats when submission
            # is via LLM API
            if self._iter_stats_result:
                self.dispatch_stats_thread.start()

            if self._iter_kv_events_result:
                self.dispatch_kv_cache_events_thread.start()

        self._handle_background_error()

    def _start_executor_workers(self, worker_kwargs):

        self_ref = weakref.ref(self)

        def mpi_done_callback(future: concurrent.futures.Future):
            # This is called when the MPI worker is done, so future.exception()
            # will not block.
            if future.exception() is not None:
                if self_ := self_ref():
                    self_._error_queue.put_nowait(future.exception())

        tracer_init_kwargs = get_tracer().init_kwargs if enable_llm_tracer(
        ) else None

        from tensorrt_llm._torch.models.modeling_auto import MODEL_CLASS_MAPPING
        torch.cuda.Stream()
        self.mpi_futures = self.mpi_session.submit(
            worker_main,
            **worker_kwargs,
            worker_cls=self.worker_cls,
            tracer_init_kwargs=tracer_init_kwargs,
            _torch_model_class_mapping=MODEL_CLASS_MAPPING,
            ready_signal=GenerationExecutorProxy.READY_SIGNAL,
            pause_signal=GenerationExecutorProxy.PAUSED_SIGNAL,
        )
        for fut in self.mpi_futures:
            fut.add_done_callback(mpi_done_callback)

        self.workers_started = True
        self.model_loaded = False

        while True:
            if self.worker_init_status_queue.poll(1):
                ready_signal, error_trace = self.worker_init_status_queue.get()
                break
            if any(fut.done() for fut in self.mpi_futures):
                logger.error("Executor worker died during initialization.")
                raise RuntimeError("Executor worker died during initialization")
            self._handle_background_error()

        # 处理可能的状态：READY (原来) 或 PAUSED_NO_MODEL（lazy 模式）
        if ready_signal == GenerationExecutorProxy.READY_SIGNAL:
            self.model_loaded = True
        elif ready_signal == GenerationExecutorProxy.PAUSED_SIGNAL:
            self.model_loaded = False
        else:
            # error case
            logger.error(f"Executor worker initialization error: {error_trace}")
            self.mpi_session.shutdown_abort(reason=ready_signal)
            raise RuntimeError("Executor worker returned error") from ready_signal

    def load_model(
        self,
        model_name: str,
        engine_dir: Union[str, Path],
        tp_size: int = 1,
        pp_size: int = 1,
        timeout: Optional[float] = None
    ) -> None:
        """
        Send LOAD_MODEL control command to the leader worker and wait for READY_SIGNAL from workers.
        """
        if getattr(self, "model_loaded", False):
            logger.info("Model already loaded on workers.")
            return

        # Make sure control_queue exists
        if not hasattr(self, "control_queue_lst") or self.control_queue_lst is None:
            raise RuntimeError("Control queue is not available to trigger model load.")

        # The signal should be broadcasted to the needed workers.
        load_model_request = LoadModelRequest(
            model_name=model_name,
            engine_dir=engine_dir,
            worker_ranks=list(range(tp_size * pp_size)),
            tp_size=tp_size,
            pp_size=pp_size,
        )
        self.model_world_size = tp_size * pp_size
        for i in range(self.model_world_size):
            self.control_queue_lst[i].put_noblock(load_model_request)
        logger.info("Sent LOAD_MODEL to workers, waiting for READY...")

        start = time.time()
        while True:
            if self.worker_init_status_queue.poll(1):
                ready_signal, error_trace = self.worker_init_status_queue.get()
                if ready_signal == GenerationExecutorProxy.READY_SIGNAL:
                    self.model_loaded = True
                    logger.info("All workers reported READY after model loading.")
                    return
                else:
                    logger.error(f"Worker returned error during model load: {error_trace}")
                    self.mpi_session.shutdown_abort(reason=ready_signal)
                    raise RuntimeError("Executor worker returned error") from ready_signal

            # timeout check
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError("Timed out waiting for workers to load model")

            # if any worker future finished unexpectedly -> error
            if any(fut.done() for fut in self.mpi_futures):
                raise RuntimeError("Executor worker died during model loading")

            self._handle_background_error()
    
    def unload_model(self):
        """
        Send UNLOAD_MODEL control command to all workers (leader and subordinates).
        """
        if not getattr(self, "model_loaded", True):
            logger.warning("Model has not been loaded on workers yet.")
            return
        
        # Send UnloadModelRequest to leader via request_queue
        self.request_queue.put(UnloadModelRequest())
        
        # # Send UnloadModelRequest to all subordinate workers via control_queue
        # # This ensures subordinates can exit from block_subordinates()
        # if hasattr(self, "control_queue_lst") and self.control_queue_lst is not None:
        #     for i in range(self.model_world_size):
        #         try:
        #             self.control_queue_lst[i].put_noblock(UnloadModelRequest())
        #         except Exception as e:
        #             logger.warning(f"Failed to send UnloadModelRequest to control_queue {i}: {e}")
        
        logger.info("Sent UNLOAD_MODEL to all workers, waiting for PAUSE...")

        while True:
            if self.worker_init_status_queue.poll(1):
                pause_signal, error_trace = self.worker_init_status_queue.get()
                if pause_signal == GenerationExecutorProxy.PAUSED_SIGNAL:
                    self.model_loaded = False
                    logger.info("All workers reported PAUSED after model unloading.")
                    return
                else:
                    logger.error(f"Worker returned error during model unload: {error_trace}")
                    self.mpi_session.shutdown_abort(reason=pause_signal)
                    raise RuntimeError("Executor worker returned error") from pause_signal
            
            # if any worker future finished unexpectedly -> error
            if any(fut.done() for fut in self.mpi_futures):
                raise RuntimeError("Executor worker died during model unloading")

            self._handle_background_error()

    def _abort_all_requests(self):
        # The results can be finished during this loop, so self._results may be changed.
        for result in list(self._results.values()):
            result.abort()

    def pre_shutdown(self):
        if not self.workers_started:
            return
        print_colored_debug('Proxy.pre_shutdown...\n', "yellow")

        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        self._abort_all_requests()

        # notify the workers to quit
        if all(not f.done() for f in self.mpi_futures):
            self.request_queue.put_noblock(None, retry=4)
            if hasattr(self, 'control_queue_lst'):
                for control_queue in self.control_queue_lst:
                    try:
                        control_queue.put_noblock(None, retry=2)
                    except Exception as e:
                        logger.warning(f"Failed to send None to control_queue: {e}")

    def shutdown(self):
        if not self.workers_started:
            return

        if not self.doing_shutdown:
            self.pre_shutdown()

        print_colored_debug('Proxy.shutdown...\n', "yellow")

        for f in self.mpi_futures:
            try:
                f.result()
            except:
                # The errors are already captured in mpi_done_callback, ignored
                # here
                pass

        # step2: notify the background threads to quit
        if self.dispatch_result_thread is not None and self.dispatch_result_thread.is_alive(
        ):
            self.dispatch_result_thread.stop()
            self.dispatch_result_thread.join()
        if self.dispatch_stats_thread is not None and self.dispatch_stats_thread.is_alive(
        ):
            self.dispatch_stats_thread.stop()
            self.dispatch_stats_thread.join()
        if self.dispatch_kv_cache_events_thread is not None and self.dispatch_kv_cache_events_thread.is_alive(
        ):
            self.dispatch_kv_cache_events_thread.stop()
            self.dispatch_kv_cache_events_thread.join()

        # step3: finish all remaining work

        # close all the sockets
        self.request_queue.close()
        self.worker_init_status_queue.close()
        self.result_queue.close()
        self.mp_stats_queue.close()
        self.kv_cache_events_queue.close()
        
        # Close control_queue in control_queue_list if it exists (lazy_load mode)
        if hasattr(self, 'control_queue_list') and self.control_queue_list is not None:
            for control_queue in self.control_queue_list:
                control_queue.close()

        self.workers_started = False
        self.mpi_session.shutdown()

        # Process the errors in-case error during shutting down the threads
        self._handle_background_error()

        if enable_llm_debug():
            print_alive_threads()

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """
            Low-level API to the executor. Return a "future" GenerationResult
            which can be waited.
            Forwards the request to the workers through the request queue.
        """
        # 加入模型未加载检查
        if not getattr(self, "model_loaded", True):
            # 如果未加载模型，拒绝请求（你也可以返回更友好的异常或 HTTP 状态）
            raise RuntimeError("Model not loaded yet. Call trigger_model_load() first.")

        self._start_dispatch_threads()

        request.set_id(self._get_next_client_id())
        logprob_params = self._get_logprob_params(request)

        result = GenerationResult(
            request,
            background_error_handler=self._handle_background_error,
            executor=self,
            disaggregated_params=request.disaggregated_params,
            logprob_params=logprob_params)
        self._results[request.id] = result

        with nvtx_range_debug("request_queue.put"):
            self.request_queue.put(request)

        self._handle_background_error()

        return result

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def get_current_load_stats(self) -> dict:
        """Get current load statistics from the worker.
        
        This method sends a LoadStatsRequest to the worker and returns a LoadResult
        object that can be used to retrieve the statistics through the dedicated stats queue.
        
        Returns:
            dict: A result object that can be used to get load statistics.
        """
        # return load_result
        self.request_queue.put(LoadStatsRequest())
        data = self.load_stats_queue.get()
        return json.loads(data)
    
    async def aget_current_load_stats(self) -> dict:
        """Asynchronously get current load statistics from the worker.
        
        This method sends a LoadStatsRequest to the worker and returns a LoadResult
        object that can be used to retrieve the statistics through the dedicated stats queue.
        
        Returns:
            LoadResult: A result object that can be used to get load statistics.
        """
        
        self.request_queue.put(LoadStatsRequest())
        data = await asyncio.to_thread(self.load_stats_queue.get)
        return json.loads(data)

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False  # propagate the exception
