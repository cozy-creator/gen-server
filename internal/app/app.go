package app

import (
	"context"
	"fmt"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/db"
	"github.com/cozy-creator/gen-server/internal/db/drivers"
	"github.com/cozy-creator/gen-server/internal/db/models"
	"github.com/cozy-creator/gen-server/internal/db/repository"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/filestorage"
	"github.com/cozy-creator/gen-server/internal/services/fileuploader"
	"github.com/cozy-creator/gen-server/pkg/ethical_filter"
	"github.com/cozy-creator/gen-server/pkg/logger"
	"github.com/uptrace/bun"
	"go.uber.org/zap"
)

type App struct {
	mq           			mq.MQ
	db           			*bun.DB
	config       			*config.Config
	ctx          			context.Context
	cancelFunc   			context.CancelFunc
	fileuploader 			*fileuploader.Uploader

	SafetyFilter			*ethical_filter.SafetyFilter
	Logger           		*zap.Logger
	
	APIKeyRepository 		repository.IAPIKeyRepository
	EventRepository 		repository.IEventRepository
	ImageRepository 		repository.IImageRepository
	// JobMetricRepository 	repository.IJobMetricRepository
	JobRepository   		repository.IJobRepository
	// PipelineDefRepository 	repository.IPipelineDefRepository
}

// Option funcs used to initialize the App struct
type OptionFunc func(app *App) error

func WithDB(driver drivers.Driver) OptionFunc {
	return func(app *App) error {
		app.db = driver.GetDB()
		return nil
	}
}

func WithLogger(logger *zap.Logger) OptionFunc {
	return func(app *App) error {
		app.Logger = logger
		return nil
	}
}

func WithMQ() OptionFunc {
    return func(app *App) error {
        mq, err := mq.NewMQ(app.config)
        if err != nil {
            return err
        }
        app.mq = mq
        return nil
    }
}

func WithDBInitialization() OptionFunc {
    return func(app *App) error {
        dbConn, err := db.NewConnection(app.ctx, app.config)
        if err != nil {
            return err
        }
        app.db = dbConn.GetDB()

        // Ensure tables exist
        err = app.db.RunInTx(app.ctx, nil, func(ctx context.Context, tx bun.Tx) error {
            tables := []interface{}{
                (*models.APIKey)(nil),
                (*models.Event)(nil),
                (*models.Image)(nil),
                (*models.Job)(nil),
                (*models.JobMetric)(nil),
                (*models.PipelineDef)(nil),
            }

            for _, table := range tables {
                if _, err := tx.NewCreateTable().
                    Model(table).
                    IfNotExists().
                    Exec(ctx); err != nil {
                    return fmt.Errorf("failed to create table: %w", err)
                }
            }
            return nil
        })
        if err != nil {
            return err
        }

        // Initialize repositories
        app.APIKeyRepository = repository.NewAPIKeyRepository(app.db)
        app.EventRepository = repository.NewEventRepository(app.db)
        app.ImageRepository = repository.NewImageRepository(app.db)
        app.JobRepository = repository.NewJobRepository(app.db)

        // Load pipeline defs from Database for enabled models
        config.LoadPipelineDefsFromDB(app.Context(), app.DB())

        return nil
    }
}

func WithFileUploader() OptionFunc {
    return func(app *App) error {
        filestorage, err := filestorage.NewFileStorage(app.Config())
        if err != nil {
            return err
        }
        app.fileuploader = fileuploader.NewFileUploader(filestorage, 10)
        return nil
    }
}

func WithSafetyFilter() OptionFunc {
	return func(app *App) error {
		if (app.config.OpenAI == nil) {
			return fmt.Errorf("openAI API-key is not set. Cannot enable safety filter")
		}
        
		filter, err := ethical_filter.NewSafetyFilter(app.config.OpenAI.APIKey)
		if err != nil {
			return err
		}

		app.SafetyFilter = filter
		return nil
	}
}

func NewApp(config *config.Config, options ...OptionFunc) (*App, error) {
    logger, err := logger.InitLogger(config)
    if err != nil {
        return nil, err
    }
    defer logger.Sync()

    ctx, cancel := context.WithCancel(context.Background())

    app := &App{
        ctx:        ctx,
        config:     config,
        Logger:     logger,
        cancelFunc: cancel,
    }

    // Apply all options
    for _, opt := range options {
        if err := opt(app); err != nil {
			// Continue even if some options fail
            app.Logger.Error("failed to apply option", zap.Error(err))
        }
    }

    return app, nil
}

func (app *App) Close() {
	app.cancelFunc()

	if app.mq != nil {
		app.mq.Close()
	}
}

func (app *App) Config() *config.Config {
	return app.config
}

func (app *App) Context() context.Context {
	return app.ctx
}

func (app *App) MQ() mq.MQ {
	return app.mq
}

func (app *App) DB() *bun.DB {
	return app.db
}

func (app *App) Uploader() *fileuploader.Uploader {
	return app.fileuploader
}

