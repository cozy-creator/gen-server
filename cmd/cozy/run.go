package cmd

import (
	"cozy-creator/gen-server/internal"
	"cozy-creator/gen-server/internal/config"
	"cozy-creator/gen-server/internal/services"
	"cozy-creator/gen-server/internal/worker"
	"fmt"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Start the cozy gen-server",
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
	runCmd.Flags().Int("port", 8881, "Port to run the server on")
	runCmd.Flags().String("host", "localhost", "Host to run the server on")
	runCmd.Flags().String("environment", "development", "Environment configuration; affects default behavior")

	viper.BindPFlag("port", runCmd.Flags().Lookup("port"))
	viper.BindPFlag("host", runCmd.Flags().Lookup("host"))
	viper.BindPFlag("environment", runCmd.Flags().Lookup("environment"))
}

func startUploadWorker() error {
	uploader, err := services.GetUploader()
	if err != nil {
		return fmt.Errorf("error getting uploader: %w", err)
	}

	worker.InitializeUploadWorker(uploader, 10)
	return nil
}
