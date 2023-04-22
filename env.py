def load_env():

    variables = {}

    with open(".env") as f:
        for line in f:
            key, value = line.split("=")
            variables[key] = value.strip()

    return variables