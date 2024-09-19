package app

import (
	"context"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/mq"
	"go.uber.org/zap"
)

type App struct {
	mq     mq.MQueue
	config *config.Config
	ctx    context.Context
	Logger *zap.Logger
}

func NewApp(ctx context.Context, config *config.Config) (*App, error) {
	logger, err := initLogger(config.Environment)
	if err != nil {
		return nil, err
	}
	defer logger.Sync()

	return &App{
		ctx:    ctx,
		config: config,
		Logger: logger,
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

func (app *App) Close() {
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
