package cmd

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/server"
	"github.com/cozy-creator/gen-server/internal/services/filestorage"
	"github.com/cozy-creator/gen-server/internal/services/generation"
	"github.com/cozy-creator/gen-server/internal/services/workflow"
	"github.com/cozy-creator/gen-server/tools"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var Cmd = &cobra.Command{
	Use:   "run",
	Short: "Start the cozy gen-server",
	RunE:  runApp,
}

func init() {
	flags := Cmd.Flags()
	flags.Int("port", 8881, "Port to run the server on")
	flags.Int("tcp-port", 8882, "Port to run the tcp server on")
	flags.String("host", "localhost", "Host to run the server on")
	flags.String("environment", "dev", "Environment configuration")
	flags.String("mq-type", "inmemory", "Message queue type: 'inmemory' or 'pulsar'")
	flags.String("warmup-models", "", "Models to be loaded and warmed up on startup")

	flags.String("db-dsn", "file:./test.db", "Database DSN (Connection URL or Path)")

	flags.String("filesystem-type", "local", "Filesystem type: 'local' or 's3'")

	flags.String("s3-access-key", "", "S3 access key")
	flags.String("s3-secret-key", "", "S3 secret key")
	flags.String("s3-region-name", "", "S3 region name")
	flags.String("s3-bucket-name", "", "S3 bucket name")
	flags.String("s3-folder", "", "S3 folder")
	flags.String("s3-public-url", "", "Public URL for S3 files")
	flags.String("s3-endpoint-url", "", "S3 endpoint URL")

	bindFlags()
	bindEnvs()
}

func bindFlags() {
	flags := Cmd.Flags()

	viper.BindPFlag("port", flags.Lookup("port"))
	viper.BindPFlag("host", flags.Lookup("host"))
	viper.BindPFlag("mq_type", flags.Lookup("mq-type"))
	viper.BindPFlag("tcp_port", flags.Lookup("tcp-port"))
	viper.BindPFlag("environment", flags.Lookup("environment"))
	viper.BindPFlag("warmup_models", flags.Lookup("warmup-models"))
	viper.BindPFlag("filesystem_type", flags.Lookup("filesystem-type"))

	// Database
	viper.BindPFlag("db.dsn", flags.Lookup("db-dsn"))

	// S3 Credentials
	viper.BindPFlag("s3.access_key", flags.Lookup("s3-access-key"))
	viper.BindPFlag("s3.secret_key", flags.Lookup("s3-secret-key"))
	viper.BindPFlag("s3.region_name", flags.Lookup("s3-region-name"))
	viper.BindPFlag("s3.bucket_name", flags.Lookup("s3-bucket-name"))
	viper.BindPFlag("s3.folder", flags.Lookup("s3-folder"))
	viper.BindPFlag("s3.vanity_url", flags.Lookup("s3-vanity-url"))
	viper.BindPFlag("s3.endpoint_url", flags.Lookup("s3-endpoint-url"))
}

func bindEnvs() {
	// External API services
	viper.BindEnv("openai.api_key", "OPENAI_API_KEY")
	viper.BindEnv("replicate.api_key", "REPLICATE_API_KEY")
	viper.BindEnv("luma.api_key", "LUMA_API_KEY")
	viper.BindEnv("runway.api_key", "RUNWAY_API_KEY")
	viper.BindEnv("bfl.api_key", "BFL_API_KEY")
	viper.BindEnv("hf_token", "HF_TOKEN")
}

func runApp(_ *cobra.Command, _ []string) error {
	defer func() {
		if err := recover(); err != nil {
			fmt.Println("Recovered from panic:", err)
		}
	}()

	var wg sync.WaitGroup
	errc := make(chan error, 4)
	signalc := make(chan os.Signal, 1)

	app, err := createNewApp()
	if err != nil {
		return err
	}
	defer app.Close()

	cfg := app.Config()
	ctx := app.Context()

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
		if err := runPythonGenServer(ctx, cfg); err != nil {
			errc <- err
		}
	}()

	go func() {
		defer wg.Done()
		if err := runGenerationProcessessor(ctx, cfg, app.MQ()); err != nil {
			errc <- err
		}
	}()

	go func() {
		defer wg.Done()
		if err := runWorkflowProcessor(app); err != nil {
			errc <- err
		}
	}()

	// go func() {
	// 	defer wg.Done()
	// 	if err := downloadEnabledModels(ctx, app.Config()); err != nil {
	// 		errc <- err
	// 	}
	// }()

	signal.Notify(signalc, os.Interrupt, syscall.SIGTERM)

	select {
	case err := <-errc:
		return err
	case <-signalc:
		server.Stop(ctx)
		return nil
	default:
		wg.Wait()
		return nil
	}
}

func createNewApp() (*app.App, error) {
	app, err := app.NewApp(config.MustGetConfig())
	if err != nil {
		return nil, err
	}

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

func runPythonGenServer(ctx context.Context, cfg *config.Config) error {
	if err := tools.StartPythonGenServer(ctx, "0.3.0", cfg); err != nil {
		return err
	}

	return nil
}

func runGenerationProcessessor(ctx context.Context, cfg *config.Config, mq mq.MQ) error {
	if err := generation.RunProcessor(ctx, cfg, mq); err != nil {
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

// func downloadEnabledModels(ctx context.Context, cfg *config.Config) error {
// 	if err := models.DownloadEnabledModels(ctx, cfg); err != nil {
// 		return err
// 	}

// 	return nil
// }

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

