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
	"github.com/cozy-creator/gen-server/pkg/logger"
	"github.com/uptrace/bun"
	"go.uber.org/zap"
)

type App struct {
	mq           	mq.MQ
	db           	*bun.DB
	config       	*config.Config
	ctx          	context.Context
	cancelFunc   	context.CancelFunc
	fileuploader 	*fileuploader.Uploader

	Logger           *zap.Logger
	JobRepository    repository.IJobRepository
	ImageRepository  repository.IImageRepository
	APIKeyRepository repository.IAPIKeyRepository
	EventRepository  repository.IEventRepository
}

type OptionFunc func(app *App)

func WithDB(driver drivers.Driver) OptionFunc {
	return func(app *App) {
		app.db = driver.GetDB()
	}
}

func WithLogger(logger *zap.Logger) OptionFunc {
	return func(app *App) {
		app.Logger = logger
	}
}

func NewApp(config *config.Config, options ...OptionFunc) (*App, error) {
	logger, err := logger.InitLogger(config)
	if err != nil {
		return nil, err
	}
	defer logger.Sync()

	ctx, cancel := context.WithCancel(context.Background())

	return &App{
		ctx:        ctx,
		config:     config,
		Logger:     logger,
		cancelFunc: cancel,
	}, nil
}

func (app *App) InitializeMQ() error {
	mq, err := mq.NewMQ(app.config)
	if err != nil {
		return err
	}

	app.mq = mq
	return nil
}

func (app *App) InitializeUploadWorker(filestorage filestorage.FileStorage) {
	app.fileuploader = fileuploader.NewFileUploader(filestorage, 10)
}

func (app *App) InitializeDB() error {
	db, err := db.NewConnection(app.ctx, app.config)
	if err != nil {
		return err
	}

	// Ensure tables exist before initializing repositories
	err = db.GetDB().RunInTx(app.ctx, nil,
		func(ctx context.Context, tx bun.Tx) error {
			tables := []interface{}{
				(*models.APIKey)(nil),
				(*models.Job)(nil),
				(*models.Image)(nil),
				(*models.Event)(nil),
			}

			for _, table := range tables {
				_, err := tx.NewCreateTable().
					Model(table).
					IfNotExists().
					Exec(ctx)
				if err != nil {
					return fmt.Errorf("failed to create table: %w", err)
				}
			}
			return nil
		})
	if err != nil {
		return err
	}

	app.db = db.GetDB()
	app.JobRepository = repository.NewJobRepository(app.db)
	app.ImageRepository = repository.NewImageRepository(app.db)
	app.EventRepository = repository.NewEventRepository(app.db)
	app.APIKeyRepository = repository.NewAPIKeyRepository(app.db)
	return nil
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

func (app *App) GetModels(ctx context.Context, modelNames []string) ([]models.Model, error) {
	return repository.GetModels(ctx, app.DB(), modelNames)
}