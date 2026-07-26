import os
from aiohttp import web

routes = web.RouteTableDef()

@routes.get("/clips/{clip_id}/detections")
async def get_detections_for_clip(request: web.Request) -> web.Response:
    repo = request.app["clips_repository"]
    lowres_size = request.app["lowres_size"]

    clip_id = request.match_info.get("clip_id")

    started_at = repo.get_clip_started_at(clip_id)
    if started_at is None:
        return web.json_response({"error": "Clip not found"}, status=404)

    rows = repo.get_detections_for_clip(clip_id)
    detections = [
        {
            "offset_seconds": (row[0] - started_at) / 1000.0,
            "class_name": row[1],
            "confidence": row[2],
            "bbox_x": row[3],
            "bbox_y": row[4],
            "bbox_width": row[5],
            "bbox_height": row[6],
        }
        for row in rows
    ]

    result = {
        "lowres_size": list(lowres_size),
        "detections": detections,
    }

    return web.json_response(result)

@routes.get("/clips/{clip_id}")
async def get_clip(request: web.Request) -> web.StreamResponse:
    repo = request.app["clips_repository"]
    clips_dir = request.app["clips_dir"]

    clip_id = request.match_info.get("clip_id")
    clip_path = f"{clips_dir}/clip_{clip_id}.mp4"

    if not os.path.exists(clip_path):
        return web.json_response({"error": "Clip not found"}, status=404)

    return web.FileResponse(clip_path)

