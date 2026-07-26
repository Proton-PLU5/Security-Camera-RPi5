from aiohttp import web

routes = web.RouteTableDef()

@routes.post("/offer")
async def offer(request: web.Request) -> web.Response:
    params = await request.json()
    manager = request.app["webrtc_manager"]
    local_desc = await manager.create_answer(params["sdp"], params["type"])
    return web.json_response({"sdp": local_desc.sdp, "type": local_desc.type})