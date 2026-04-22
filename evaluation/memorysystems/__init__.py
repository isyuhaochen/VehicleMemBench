from . import lightmem, mem0, memobase, memorybank, memos, supermemory


SYSTEM_MODULES = {
    "mem0": mem0,
    "memos": memos,
    "lightmem": lightmem,
    "supermemory": supermemory,
    "memobase": memobase,
    "memorybank": memorybank,
}

SUPPORTED_MEMORY_SYSTEMS = tuple(SYSTEM_MODULES.keys())


def get_system_module(name: str):
    try:
        return SYSTEM_MODULES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported memory system: {name}. Supported: {', '.join(SUPPORTED_MEMORY_SYSTEMS)}"
        ) from exc
