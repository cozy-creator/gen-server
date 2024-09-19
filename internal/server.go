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
	inner  *http.Server
	engine *gin.Engine
	app    *app.App
}

func NewServer(app *app.App) *Server {
	cfg := app.GetConfig()

	return &Server{
		Port: cfg.Port,
		Host: cfg.Host,
		app:  app,
	}
}

func (s *Server) SetupEngine() error {
	r := gin.New()
	cfg := s.app.GetConfig()

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

	httpServer := &http.Server{
		Addr:    fmt.Sprintf("%s:%d", s.Host, s.Port),
		Handler: r,
	}

	s.inner = httpServer
	s.engine = r

	return nil
}

func (s *Server) Start() (err error) {
	if s.engine == nil {
		return fmt.Errorf("engine is not initialized")
	}

	if err := s.inner.ListenAndServe(); err != nil {
		return err
	}
	return nil
}

func (s *Server) Stop(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()

	if err := s.inner.Shutdown(ctx); err != nil {
		return err
	}

	return nil
}

func (s *Server) GetEngine() *gin.Engine {
	return s.engine
}

func getGinMode(env string) string {
	switch env {
	case "development":
		return gin.DebugMode
	case "test":
		return gin.TestMode
	case "production":
		return gin.ReleaseMode
	default:
		return gin.ReleaseMode
	}
}
