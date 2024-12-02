package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/server"
	"github.com/cozy-creator/gen-server/internal/services/filestorage"
	"github.com/cozy-creator/gen-server/internal/services/generation"
	"github.com/cozy-creator/gen-server/internal/services/model_downloader"
	"github.com/cozy-creator/gen-server/internal/services/workflow"
	"github.com/cozy-creator/gen-server/tools"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var Cmd = &cobra.Command{
	Use:   "run",
	Short: "Start the cozy gen-server",
	// PreRunE: bindFlags,
	RunE:  runApp,
}

func init() {
	cobra.OnInitialize(initDefaults)
	flags := Cmd.Flags()

	flags.Int("port", 8881, "Port to run the server on")
	flags.String("host", "localhost", "Host to run the server on")
	flags.String("environment", "dev", "Environment configuration")
	flags.Bool("disable-auth", false, "Disable authentication when receiving requests")
	flags.StringSlice("warmup-models", []string{}, "Models to be loaded and warmed up on startup")
	flags.String("filesystem-type", "local", "Filesystem type: 'local' or 's3'")
	flags.String("public-dir", "", "Path where static files should be served from. Relative paths are relative to the current working directory, not the location of the gen-server executable.")

	flags.String("db-dsn", "", "Database DSN (Connection URL or Path)")
	flags.String("pulsar-url", "", "URL of the pulsar broker. Example: pulsar+ssl://my-cluster.streamnative.cloud:6651")

	viper.BindPFlags(flags)

	// These have to be bound manually due to nesting and hyphens
	viper.BindPFlag("db.dsn", flags.Lookup("db-dsn"))
	viper.BindPFlag("pulsar.url", flags.Lookup("pulsar-url"))
	viper.BindPFlag("public_dir", flags.Lookup("public-dir"))

	bindEnvs()
}

func bindEnvs() {
	// Core settings (will use COZY_ prefix)
	// Example: COZY_PORT
    viper.BindEnv("port")
    viper.BindEnv("host")
    viper.BindEnv("environment")
    viper.BindEnv("disable_auth")
    viper.BindEnv("warmup_models")
    viper.BindEnv("filesystem_type")
    viper.BindEnv("public_dir")

	viper.BindEnv("db.dsn")
	viper.BindEnv("pulsar.url")

	// S3 environment bindings (will automatically use COZY_ prefix)
	// example: COZY_S3_ACCESS_KEY
	viper.BindEnv("s3.access_key")
	viper.BindEnv("s3.secret_key")
	viper.BindEnv("s3.region_name")
	viper.BindEnv("s3.bucket_name")
	viper.BindEnv("s3.folder")
	viper.BindEnv("s3.public_url")
	viper.BindEnv("s3.endpoint_url")

	// External API services (does NOT use COZY_ prefix)
	viper.BindEnv("openai.api_key", "OPENAI_API_KEY")
	viper.BindEnv("replicate.api_key", "REPLICATE_API_KEY")
	viper.BindEnv("luma_ai.api_key", "LUMA_API_KEY")
	viper.BindEnv("runway.api_key", "RUNWAY_API_KEY")
	viper.BindEnv("bfl.api_key", "BFL_API_KEY")
	viper.BindEnv("hf_token", "HF_TOKEN")
	viper.BindEnv("civitai.api_key", "CIVITAI_API_KEY")
}

// Initialize defaults that depend upon other environment variables being initialized first
func initDefaults() {
	viper.SetDefault("db.dsn", "file:" + filepath.Join(viper.GetString("cozy_home"), "data", "main.db"))
	viper.SetDefault("public_dir", server.GetDefaultPublicDir(viper.GetString("environment")))
}

func runApp(_ *cobra.Command, _ []string) error {
	// defer func() {
	// 	if err := recover(); err != nil {
	// 		fmt.Println("Recovered from panic:", err)
	// 	}
	// }()

	// create base context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	errc := make(chan error, 4)
	signalc := make(chan os.Signal, 1)

	app, err := createNewApp()
	if err != nil {
		return err
	}
	defer app.Close()

	downloaderManager, err := model_downloader.NewModelDownloaderManager(app)
	if err != nil {
        return fmt.Errorf("failed to initialize model downloader manager: %w", err)
    }

	cfg := app.Config()

	wg.Add(4)

	server, err := runServer(app)
	if err != nil {
		if errors.Is(err, http.ErrServerClosed) {
			fmt.Println("Server stopped successfully")
		}

		return err
	}

	go func() {
		defer wg.Done()
		if err := runPythonRuntime(ctx, cfg); err != nil {
			errc <- err
		}
	}()

	go func() {
		defer wg.Done()
		if err := runGenerationProcessessor(ctx, cfg, app.MQ(), app, downloaderManager); err != nil {
			errc <- err
		}
	}()

	go func() {
		defer wg.Done()
		if err := runWorkflowProcessor(app); err != nil {
			errc <- err
		}
	}()

	go func() {
		defer wg.Done()
		if err := downloadEnabledModels(app, downloaderManager); err != nil {
			if !errors.Is(err, context.Canceled) {
				errc <- fmt.Errorf("model download error: %w", err)
			}
		}
	}()

	signal.Notify(signalc, os.Interrupt, syscall.SIGTERM)

	shutdownComplete := make(chan struct{})

	// wait for shutdown signal
	go func() {
		<-signalc
		fmt.Println("\nReceived shutdown signal. Starting graceful shutdown...")

		cancel()

		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer shutdownCancel()

		if err := server.Stop(shutdownCtx); err != nil && !errors.Is(err, context.Canceled) {
            errc <- fmt.Errorf("error stopping server: %w", err)
            return
        }

		// wait for all gorutines with timeout
		done := make(chan struct{})
		go func() {
			wg.Wait()
			close(done)
		}()

		select {
		case <-done:
			fmt.Println("All services stopped successfully")
		case <-shutdownCtx.Done():
			fmt.Println("Warning: Shutdown timed out, some services may not have stopped cleanly")
		}
		close(shutdownComplete)
	}()

	// wait for shutdown to complete or error
	select {
	case err := <-errc:
		cancel()
		if errors.Is(err, context.Canceled) {
			return nil
		}
		return err
	case <-ctx.Done():
		select {
		case <-shutdownComplete:
			return nil
		case <-time.After(35 * time.Second):
			return nil // Don't return error on timeout
		case err := <-errc:
			if errors.Is(err, context.Canceled) {
				return nil
			}
			return err
		case <-signalc:
			fmt.Println("\nForced shutdown initiated")
			return nil
		}
	}
}

func createNewApp() (*app.App, error) {
	app, err := app.NewApp(config.MustGetConfig())
	if err != nil {
		return nil, err
	}

	// TO DO: remove this block
	configJSON, err := json.MarshalIndent(app.Config(), "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal config: %w", err)
	}
	fmt.Printf("Finalized config: %s\n", configJSON)

	if err := app.InitializeMQ(); err != nil {
		fmt.Println("Error initializing MQ: ", err)
		return nil, err
	}

	if err := app.InitializeDB(); err != nil {
		return nil, err
	}

	filestorage, err := filestorage.NewFileStorage(app.Config())
	if err != nil {
		return nil, err
	}

	app.InitializeUploadWorker(filestorage)

	return app, nil
}

func runPythonRuntime(ctx context.Context, cfg *config.Config) error {
	if err := tools.StartPythonRuntime(ctx, cfg); err != nil {
		fmt.Println(err)
		return err
	}

	return nil
}

func runGenerationProcessessor(ctx context.Context, cfg *config.Config, mq mq.MQ, app *app.App, downloader *model_downloader.ModelDownloaderManager) error {
	if err := generation.RunProcessor(ctx, cfg, mq, app, downloader); err != nil {
		return err
	}

	return nil
}

func runWorkflowProcessor(app *app.App) error {
	if err := workflow.RunProcessor(app); err != nil {
		return err
	}

	return nil
}

func downloadEnabledModels(app *app.App, downloaderManager *model_downloader.ModelDownloaderManager) error {

	ctx := app.Context()

	errChan := make(chan error, 1)
	go func() {
		errChan <- downloaderManager.InitializeModels()
	}()

	select {
	case <-ctx.Done():
		return context.Canceled
	case err := <-errChan:
		return err
	}
}

func runServer(app *app.App) (*server.Server, error) {
	server, err := server.NewServer(app.Config())
	if err != nil {
		return nil, err
	}

	// Setup the server routes
	server.SetupRoutes(app)

	errc := make(chan error, 1)
	go func() {
		fmt.Printf("Gen-Server Started on Port %v\n", app.Config().Port)
		errc <- server.Start()
	}()

	select {
	case err := <-errc:
		return nil, err
	default:
		return server, nil
	}
}
