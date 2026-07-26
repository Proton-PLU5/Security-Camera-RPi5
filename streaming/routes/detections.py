from aiohttp import web
import asyncio
import logging

routes = web.RouteTableDef()

@routes.get("/websocket/detections")
async def handle_detection_websocket(request: web.Request) -> web.WebSocketResponse | web.Response:
    detection_buffer = request.app["detection_buffer"]
    stop_event = request.app["stop_event"]

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    last_version = -1
    try:
        while not stop_event.is_set() and not ws.closed:
            detections, version = detection_buffer.get_with_version()
            if version != last_version:
                last_version = version
                await ws.send_json(detections)
            await asyncio.sleep(0.1)
    except Exception as e:
        logging.exception("Detection websocket closed unexpectedly", exc_info=e)
    finally:
        await ws.close()

    return ws