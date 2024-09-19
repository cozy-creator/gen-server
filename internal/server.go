package internal

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/cozy-creator/gen-server/internal/app"

	"github.com/gin-contrib/cors"
	"github.com/gin-contrib/logger"
	"github.com/gin-contrib/static"
	"github.com/gin-gonic/gin"
)

type Server struct {
	Port   int
	Host   string
	engine *gin.Engine
	inner  *http.Server
}

func NewServer(app *app.App) (*Server, error) {
	r := gin.New()
	cfg := app.GetConfig()

	// Set gin mode
	gin.SetMode(getGinMode(cfg.Environment))

	// Setup logger middleware
	r.Use(logger.SetLogger(
		logger.WithUTC(true),
		logger.WithSkipPath([]string{}),
	))

	// Setup CORS middleware
	r.Use(cors.New(
		cors.Config{
			AllowOrigins:     []string{"*"},
			AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
			AllowHeaders:     []string{"Origin", "Content-Length", "Content-Type", "Authorization"},
			ExposeHeaders:    []string{"Content-Length"},
			AllowCredentials: true,
			MaxAge:           300,
		},
	))

	// Serve static files
	r.Use(static.Serve("/", static.LocalFile("./web/dist", true)))
	r.Use(gin.Recovery())

	return &Server{
		engine: r,
		inner: &http.Server{
			Handler: r,
			Addr:    fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
		},
	}, nil
}

func (s *Server) Start() (err error) {
	if err := s.inner.ListenAndServe(); err != nil {
		return err
	}

	return nil
}

func (s *Server) Stop(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()

	if err := s.inner.Shutdown(ctx); err != nil {
		return err
	}

	return nil
}

func getGinMode(env string) string {
	switch env {
	case "development":
		return gin.DebugMode
	case "test":
		return gin.TestMode
	default:
		return gin.ReleaseMode
	}
}
