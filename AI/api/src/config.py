import os


class ProdConfig:
    # Database configuration
    API_TOKEN = os.environ.get("PROD_MARKET_STACK_API_KEY_SECRET")


class DevConfig:
    # Database configuration
    API_TOKEN = os.environ.get("API_KEY_SECRET")


class TestConfig:
    # Database configuration
    API_TOKEN = os.environ.get("MARKET_STACK_API_KEY")


config = {"dev": DevConfig, "test": TestConfig, "prod": ProdConfig}
