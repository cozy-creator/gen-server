package cmd

import (
	"cozy-creator/gen-server/internal/config"
	"cozy-creator/gen-server/internal/utils"
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

const (
	DefultWorkspacePath = "~/.cozy-creator"
)

var rootCmd = &cobra.Command{
	Use:   "cozy",
	Short: "Cozy Creator CLI",
	Long:  "A generative AI engine that allows you to create and run generative AI models on your own computer or in the cloud",
}

func init() {
	cobra.OnInitialize(onCommandInit)

	workspacePath, err := utils.ExpandPath(DefultWorkspacePath)
	fmt.Println("wwwww", workspacePath)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	rootCmd.PersistentFlags().String("workspace-path", workspacePath, "Path to the workspace directory")
	rootCmd.PersistentFlags().String("config-file", "", "Path to the config file")
	rootCmd.PersistentFlags().String("env-file", "", "Path to the env file")
	viper.BindPFlag("workspace_path", rootCmd.PersistentFlags().Lookup("workspace-path"))
	viper.BindPFlag("config_file", rootCmd.PersistentFlags().Lookup("config-file"))
	viper.BindPFlag("env_file", rootCmd.PersistentFlags().Lookup("env-file"))

	rootCmd.AddCommand(runCmd, downloadCmd)
	rootCmd.CompletionOptions.HiddenDefaultCmd = true
}

func onCommandInit() {
	err := config.InitConfig()
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func GetRootCmd() *cobra.Command {
	return rootCmd
}

// type CLI struct {
// 	cfg *config.Config
// }

// func NewCLI() *CLI {
// 	return &CLI{
// 		cfg: &config.Config{},
// 	}
// }

// func (c *CLI) Run() error {
// 	if len(os.Args) < 2 {
// 		return fmt.Errorf("command is required")
// 	}

// 	cmd := os.Args[1]
// 	switch cmd {
// 	case CommandStart:
// 		return c.runStartCommand(os.Args[2:])
// 	default:
// 		return fmt.Errorf("unsupported command: %s", cmd)
// 	}
// }

// func (c *CLI) runStartCommand(args []string) error {
// 	runCmd := flag.NewFlagSet(CommandStart, flag.ExitOnError)

// 	configFile := runCmd.String("config-file", "", "Path to the config file")
// 	port := runCmd.Int("port", 0, "Port to run the server on")
// 	host := runCmd.String("host", "", "Host to run the server on")
// 	environment := runCmd.String("environment", "development", "Environment to run the server in")

// 	if err := runCmd.Parse(args); err != nil {
// 		return fmt.Errorf("failed to parse %s command: %w", CommandStart, err)
// 	}

// 	cfg, err := c.loadConfig(*configFile)
// 	if err != nil {
// 		return err
// 	}

// 	// Override config values with flags if they are set
// 	if *port != 0 {
// 		cfg.Port = *port
// 	}
// 	if *host != "" {
// 		cfg.Host = *host
// 	}
// 	cfg.Environment = *environment

// 	return start(cfg)
// }

// func (c *CLI) loadConfig(configFile string) (*config.Config, error) {
// 	if configFile == "" {
// 		return c.cfg, nil
// 	}

// 	cfg, err := config.NewConfigFromFile(configFile)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to load config from file %s: %w", configFile, err)
// 	}
// 	return cfg, nil
// }