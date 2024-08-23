package cmd

import (
	"cozy-creator/go-cozy/internal"
	"cozy-creator/go-cozy/internal/config"
	"cozy-creator/go-cozy/internal/services"
	"cozy-creator/go-cozy/internal/worker"
	"fmt"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start the cozy server",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg := config.GetConfig()
		server := internal.NewHTTPServer(cfg)
		if err := server.SetupEngine(cfg); err != nil {
			return fmt.Errorf("error setting up engine: %w", err)
		}

		go startUploadWorker()

		server.SetupRoutes()
		server.Start()

		return nil
	},
}

func init() {
	startCmd.Flags().Int("port", 8881, "Port to run the server on")
	startCmd.Flags().String("host", "0.0.0.0", "Host to run the server on")
	startCmd.Flags().String("environment", "development", "Environment to run the server in")

	viper.BindPFlag("port", startCmd.Flags().Lookup("port"))
	viper.BindPFlag("host", startCmd.Flags().Lookup("host"))
	viper.BindPFlag("environment", startCmd.Flags().Lookup("environment"))
}

func startUploadWorker() error {
	uploader, err := services.GetUploader()
	if err != nil {
		return fmt.Errorf("error getting uploader: %w", err)
	}

	worker.InitializeUploadWorker(uploader, 10)
	return nil
}
