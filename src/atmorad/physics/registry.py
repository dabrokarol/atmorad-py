REFLECTION_MODELS = {}
SCATTERING_MODELS = {}

def register_reflection(name: str):
    def wrapper(cls):
        REFLECTION_MODELS[name] = cls
        return cls
    return wrapper

def register_scattering(name: str):
    def wrapper(cls):
        SCATTERING_MODELS[name] = cls
        return cls
    return wrapper