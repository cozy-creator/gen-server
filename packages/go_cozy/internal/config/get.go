package config

var cfg *Config

func GetConfig() *Config {
	if cfg == nil {
		panic("config not initialized")
	}

	return cfg
}

func SetConfig(c *Config) {
	if c == nil {
		panic("config is nil")
	}

	cfg = c
}
