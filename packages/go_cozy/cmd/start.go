package cmd

import (
	"cozy-creator/go-cozy/internal"
	"cozy-creator/go-cozy/internal/config"
	"fmt"
)

func start(cfg *config.Config) error {
	config.SetConfig(cfg)

	server := internal.NewHTTPServer(cfg)
	if err := server.SetupEngine(cfg); err != nil {
		return fmt.Errorf("error setting up engine: %w", err)
	}

	server.SetupRouter()
	server.Start()
	return nil
}
