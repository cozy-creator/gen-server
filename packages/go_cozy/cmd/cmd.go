package cmd

import (
	"cozy-creator/go-cozy/internal/config"
	"flag"
	"fmt"
	"os"
)

const (
	CommandStart = "start"
)

type CLI struct {
	cfg *config.Config
}

func NewCLI() *CLI {
	return &CLI{
		cfg: &config.Config{},
	}
}

func (c *CLI) Run() error {
	if len(os.Args) < 2 {
		return fmt.Errorf("command is required")
	}

	cmd := os.Args[1]
	switch cmd {
	case CommandStart:
		return c.runStartCommand(os.Args[2:])
	default:
		return fmt.Errorf("unsupported command: %s", cmd)
	}
}

func (c *CLI) runStartCommand(args []string) error {
	startCmd := flag.NewFlagSet(CommandStart, flag.ExitOnError)

	configFile := startCmd.String("config-file", "", "Path to the config file")
	port := startCmd.Int("port", 0, "Port to run the server on")
	host := startCmd.String("host", "", "Host to run the server on")
	environment := startCmd.String("environment", "development", "Environment to run the server in")

	if err := startCmd.Parse(args); err != nil {
		return fmt.Errorf("failed to parse %s command: %w", CommandStart, err)
	}

	cfg, err := c.loadConfig(*configFile)
	if err != nil {
		return err
	}

	// Override config values with flags if they are set
	if *port != 0 {
		cfg.Port = *port
	}
	if *host != "" {
		cfg.Host = *host
	}
	cfg.Environment = *environment

	return start(cfg)
}

func (c *CLI) loadConfig(configFile string) (*config.Config, error) {
	if configFile == "" {
		return c.cfg, nil
	}

	cfg, err := config.NewConfigFromFile(configFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load config from file %s: %w", configFile, err)
	}
	return cfg, nil
}
