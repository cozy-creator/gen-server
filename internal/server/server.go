package server

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/gin-contrib/cors"
	"github.com/gin-contrib/logger"
	"github.com/gin-contrib/static"
	"github.com/gin-gonic/gin"
)

type Server struct {
	listenAddr string
	ginEngine  *gin.Engine
	inner      *http.Server
}

func NewServer(config *config.Config) (*Server, error) {
	gin.SetMode(getGinMode(config.Environment))
	r := gin.New()

	// Setup logger middleware
	r.Use(logger.SetLogger(
		logger.WithUTC(true),
		logger.WithSkipPath([]string{}),
	))

	// Setup CORS middleware
	r.Use(cors.New(
		cors.Config{
			AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
			AllowOrigins:     []string{"*"},
			AllowHeaders:     []string{"*"},
			ExposeHeaders:    []string{"*"},
			AllowCredentials: true,
			MaxAge:           300,
		},
	))

	// Serve static files
	staticPath := config.PublicDir
	// if (staticPath == "") {
	// 	staticPath = GetDefaultPublicDir(config.Environment)
	// }
	r.Use(static.Serve("/", static.LocalFile(staticPath, true)))
	r.Use(gin.Recovery())

	return &Server{
		listenAddr: fmt.Sprintf("%s:%d", config.Host, config.Port),
		ginEngine: r,
		inner: &http.Server{
			Handler: r,
			Addr:    fmt.Sprintf("%s:%d", config.Host, config.Port),
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

	fmt.Println("Stopping server...")

	if err := s.inner.Shutdown(ctx); err != nil {
		return err
	}

	return nil
}

func getGinMode(env string) string {
	switch env {
	case "dev":
		return gin.DebugMode
	case "test":
		return gin.TestMode
	default:
		return gin.ReleaseMode
	}
}

func GetDefaultPublicDir(env string) string {
    switch env {
    case "prod":
        return "/srv/www/cozy/dist"
    default:
        return "./web/dist"
	}
}
