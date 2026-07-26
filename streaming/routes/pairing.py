import os
from aiohttp import web

routes = web.RouteTableDef()

@routes.post("/pair/token")
async def handle_pair_token(request: web.Request) -> web.Response:
    pairing_secret = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if not pairing_secret:
        return web.json_response({"error": "Missing pairing secret"}, status=400)

    # Generate a new persistent pairing token for the device
    authenticator = request.app["authenticator"]
    token = authenticator.generate_session_token(pairing_secret)

    return web.json_response({"token": token}, status=200)

@routes.get("/pair")
async def handle_pair(request: web.Request) -> web.Response:
    pairing_mode = request.app["pairing_mode"]

    if not pairing_mode.value:
        return web.json_response({"error": "Pairing mode is disabled"}, status=403)

    authenticator = request.app["authenticator"]
    
    # Generate a new persistent pairing token for the device
    pairing_secret = authenticator.generate_pairing_secret()

    return web.json_response({"pairing_secret": pairing_secret})

@routes.get("/pair/cert")
async def handle_cert(request: web.Request) -> web.FileResponse | web.Response:
    if not os.path.exists("cert.pem"):
        return web.json_response({"error": "Cert not available"}, status=500)
    
    return web.FileResponse("cert.pem")
