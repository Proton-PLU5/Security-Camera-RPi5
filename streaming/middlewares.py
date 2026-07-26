from aiohttp import web

PUBLIC_ROUTES = {"/pair", "/pair/cert", "/pair/token", "/"}

@web.middleware
async def auth_middleware(request : web.Request, handler):
    if request.path in PUBLIC_ROUTES:
        return await handler(request)

    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if not token:
        return web.json_response({"error": "Missing Authorization header"}, status=401)

    authenticator = request.app["authenticator"]
    if not authenticator.validate_session_token(token):
        return web.json_response({"error": "Invalid or expired token"}, status=401)

    return await handler(request)