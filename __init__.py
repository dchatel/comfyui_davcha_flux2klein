from comfy_api.latest import io, ComfyExtension
from .nodes import *

class DavchaFlux2KleinExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DavchaKleinCLIPLoader,
        ]

async def comfy_entrypoint() -> DavchaFlux2KleinExtension:
    return DavchaFlux2KleinExtension()
