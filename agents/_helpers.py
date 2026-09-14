import asyncio
import threading


def run_inew_loop(loop, task, results):
    result = loop.run_until_complete(task)
    results.append(result)
    return result


def run_coro(tasks):
    loop = asyncio.new_event_loop()
    threads = []
    results = []
    for task in tasks:
        thread = threading.Thread(
            target=run_inew_loop, args=[loop, task, results]
        )
        threads.append(thread)
    for thread in threads:
        thread.start()
        thread.join()
    return results
