package app

import (
	"context"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/filehandler"
	"go.uber.org/zap"
)

type App struct {
	mq          mq.MQueue
	config      *config.Config
	ctx         context.Context
	cancelFunc  context.CancelFunc
	filehandler filehandler.FileHandler

	Logger *zap.Logger
}

func NewApp(config *config.Config) (*App, error) {
	ctx, cancel := context.WithCancel(context.Background())

	logger, err := initLogger(config.Environment)
	if err != nil {
		return nil, err
	}
	defer logger.Sync()

	return &App{
		ctx:        ctx,
		config:     config,
		Logger:     logger,
		cancelFunc: cancel,
	}, nil
}

func (app *App) GetConfig() *config.Config {
	return app.config
}

func (app *App) GetContext() context.Context {
	return app.ctx
}

func (app *App) SetMq(mq mq.MQueue) {
	app.mq = mq
}

func (app *App) InitializeFileHandler() error {
	filehandler, err := filehandler.NewFileHandler(app.config)
	if err != nil {
		return err
	}

	app.filehandler = filehandler
	return nil
}

func (app *App) Close() {
	app.cancelFunc()
	// fmt.Println("App closed")

	// err := app.ctx.Err()
	// fmt.Println("App context error:", err)

	if app.mq != nil {
		app.mq.Close()
	}
}

func initLogger(env string) (*zap.Logger, error) {
	if env == "production" {
		return zap.NewProduction()
	}

	return zap.NewDevelopment()
}
