package internal

import (
	"cozy-creator/go-cozy/internal/config"
	"fmt"

	"github.com/gin-contrib/cors"
	"github.com/gin-contrib/logger"
	"github.com/gin-contrib/static"
	"github.com/gin-gonic/gin"
)

type HTTPServer struct {
	Port   int
	Host   string
	engine *gin.Engine
}

func NewHTTPServer(cfg *config.Config) *HTTPServer {
	return &HTTPServer{
		Port: cfg.Port,
		Host: cfg.Host,
	}
}

func (s *HTTPServer) SetupEngine(cfg *config.Config) error {
	r := gin.New()

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

	s.engine = r

	return nil
}

func (s *HTTPServer) Start() (err error) {
	if s.engine == nil {
		return fmt.Errorf("engine is not initialized")
	}

	err = s.engine.Run(fmt.Sprintf("%s:%d", s.Host, s.Port))
	return
}

func (s *HTTPServer) GetEngine() *gin.Engine {
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
