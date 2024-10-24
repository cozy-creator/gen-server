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

	"github.com/cozy-creator/gen-server/internal/model"
	"github.com/cozy-creator/gen-server/pkg/tcpclient"
	"github.com/cozy-creator/gen-server/internal/mq"
)

type Server struct {
	listenAddr string
	ginEngine  *gin.Engine
	inner      *http.Server
	modelManager *model.ModelManager
}

func NewServer(cfg *config.Config) (*Server, error) {

	tcpClient, err := tcpclient.NewTCPClient(
        fmt.Sprintf("%s:%d", cfg.Host, cfg.TcpPort),
        time.Duration(500)*time.Second,
        1,
    )
	
    if err != nil {
        return nil, fmt.Errorf("failed to create TCP client: %w", err)
    }

    mqInstance, err := mq.NewInMemoryMQ(10)
    if err != nil {
        return nil, fmt.Errorf("failed to create InMemoryMQ: %w", err)
    }

    modelManager := model.NewModelManager(tcpClient, mqInstance)
	
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
	if cfg.Environment == "production" {
		r.Use(static.Serve("/", static.LocalFile("/srv/www/cozy/dist", true)))
	} else {
		r.Use(static.Serve("/", static.LocalFile("./web/dist", true)))
	}

	r.Use(gin.Recovery())

	// // Initialize TCP client for model manager
    // tcpClient, err := tcpclient.NewTCPClient(
    //     fmt.Sprintf("%s:%d", cfg.Host, cfg.TcpPort),
    //     time.Duration(500)*time.Second,
    //     1,
    // )
    // if err != nil {
    //     return nil, fmt.Errorf("failed to create TCP client: %w", err)
    // }

    // modelManager := model.NewModelManager(tcpClient, nil) // or pass your MQ instance

	return &Server{
		modelManager: modelManager,
		ginEngine: r,
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

	fmt.Println("Stopping server...")

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
