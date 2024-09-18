package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/cozy-creator/gen-server/internal"
	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/services/filehandler"
	"github.com/cozy-creator/gen-server/internal/worker"
	"github.com/cozy-creator/gen-server/tools"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.uber.org/zap"
)

var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Start the cozy gen-server",
	RunE:  runServer,
}

func init() {
	initFlags()
}

func initFlags() {
	flags := runCmd.Flags()
	flags.Int("port", 9009, "Port to run the server on")
	flags.String("host", "localhost", "Host to run the server on")
	flags.Int("tcp-port", 9008, "Port to run the tcp server on")
	flags.String("environment", "development", "Environment configuration")
	flags.String("models-dir", "", "Directory for models (default: {home}/models)")
	flags.String("aux-models-dirs", "", "Additional directories for model files")
	flags.String("temp-dir", "", "Directory for temporary files (default: {home}/temp)")
	flags.String("assets-dir", "", "Directory for assets (default: {home}/assets)")
	flags.String("enabled-models", "", "Models to be downloaded and made available")
	flags.String("filesystem-type", "local", "Filesystem type: 'local' or 's3'")
	flags.String("s3-access-key", "", "S3 access key")
	flags.String("s3-secret-key", "", "S3 secret key")
	flags.String("s3-region-name", "", "S3 region name")
	flags.String("s3-bucket-name", "", "S3 bucket name")
	flags.String("s3-folder", "", "S3 folder")
	flags.String("s3-public-url", "", "Public URL for S3 files")

	bindFlags()
}

func bindFlags() {
	flags := runCmd.Flags()

	viper.BindPFlag("port", flags.Lookup("port"))
	viper.BindPFlag("host", flags.Lookup("host"))
	viper.BindPFlag("tcp_port", flags.Lookup("tcp-port"))
	viper.BindPFlag("environment", flags.Lookup("environment"))
	viper.BindPFlag("models_path", flags.Lookup("models-path"))
	viper.BindPFlag("assets_path", flags.Lookup("assets-path"))
	viper.BindPFlag("enabled_models", flags.Lookup("enabled-models"))
	viper.BindPFlag("filesystem_type", flags.Lookup("filesystem-type"))
	viper.BindPFlag("aux_models_paths", flags.Lookup("aux-models-paths"))

	// S3 Credentials
	viper.BindPFlag("s3.access_key", flags.Lookup("s3-access-key"))
	viper.BindPFlag("s3.secret_key", flags.Lookup("s3-secret-key"))
	viper.BindPFlag("s3.region_name", flags.Lookup("s3-region-name"))
	viper.BindPFlag("s3.bucket_name", flags.Lookup("s3-bucket-name"))
	viper.BindPFlag("s3.folder", flags.Lookup("s3-folder"))
	viper.BindPFlag("s3.public_url", flags.Lookup("s3-public-url"))
}

func runServer(_ *cobra.Command, _ []string) error {
	cfg := config.GetConfig()
	logger, err := initLogger(cfg.Environment)
	if err != nil {
		return fmt.Errorf("failed to initialize logger: %w", err)
	}
	defer logger.Sync()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	errChan := make(chan error, 1)

	server := internal.NewHTTPServer(cfg)

	wg.Add(4)
	go runUploadWorker(&wg, errChan, logger)
	go runPythonGenServer(ctx, &wg, errChan, cfg, logger)
	go runGenerationWorker(ctx, &wg, errChan, logger)
	go handleSignals(ctx, cancel, &wg, logger)

	go runHTTPServer(server, errChan, logger)

	select {
	case err := <-errChan:
		cancel()
		return fmt.Errorf("server error: %w", err)
	case <-ctx.Done():
		logger.Info("Shutting down gracefully...")
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer shutdownCancel()
		if err := server.Stop(shutdownCtx); err != nil {
			logger.Error("Error during server shutdown", zap.Error(err))
		}
	}

	wg.Wait()
	return nil
}

func initLogger(env string) (*zap.Logger, error) {
	if env == "production" {
		return zap.NewProduction()
	}
	return zap.NewDevelopment()
}

func runUploadWorker(wg *sync.WaitGroup, errChan chan<- error, logger *zap.Logger) {
	defer wg.Done()
	uploader, err := filehandler.GetFileHandler()
	if err != nil {
		logger.Error("Failed to get uploader", zap.Error(err))
		errChan <- fmt.Errorf("error getting uploader: %w", err)
		return
	}
	worker.InitializeUploadWorker(uploader, 10)
	logger.Info("Upload worker started")
}

func runPythonGenServer(ctx context.Context, wg *sync.WaitGroup, errChan chan<- error, cfg *config.Config, logger *zap.Logger) {
	defer wg.Done()
	if err := tools.StartPythonGenServer(ctx, "0.2.2", cfg); err != nil {
		logger.Error("Failed to start Python Gen Server", zap.Error(err))
		errChan <- fmt.Errorf("error starting Python Gen Server: %w", err)
	}
}

func runGenerationWorker(ctx context.Context, wg *sync.WaitGroup, errChan chan<- error, logger *zap.Logger) {
	defer wg.Done()
	if err := worker.StartGeneration(ctx); err != nil {
		logger.Error("Failed to start generation worker", zap.Error(err))
		errChan <- fmt.Errorf("error starting generation worker: %w", err)
	}
}

func handleSignals(ctx context.Context, cancel context.CancelFunc, wg *sync.WaitGroup, logger *zap.Logger) {
	defer wg.Done()
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, os.Interrupt, syscall.SIGTERM)

	select {
	case <-signalChan:
		logger.Info("Received shutdown signal")
		cancel()
	case <-ctx.Done():
	}
}

func runHTTPServer(server *internal.HTTPServer, errChan chan<- error, logger *zap.Logger) {
	cfg := config.GetConfig()
	if err := server.SetupEngine(cfg); err != nil {
		logger.Error("Failed to set up server engine", zap.Error(err))
		errChan <- fmt.Errorf("error setting up engine: %w", err)
		return
	}

	server.SetupRoutes()
	if err := server.Start(); err != nil {
		logger.Error("Server error", zap.Error(err))
		errChan <- fmt.Errorf("server error: %w", err)
	}
}
