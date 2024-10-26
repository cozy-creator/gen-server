package api

import (
    "net/http"
    "github.com/gin-gonic/gin"
    "github.com/cozy-creator/gen-server/internal/app"
    "github.com/cozy-creator/gen-server/internal/services/generation"
    
)

type ModelRequest struct {
    ModelIDs []string `json:"model_ids"`
    Priority bool     `json:"priority"`
}

func LoadModels(c *gin.Context) {
    var req ModelRequest
    if err := c.BindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
        return
    }

    app := c.MustGet("app").(*app.App)
    
    if err := generation.LoadModels(c.Request.Context(), app.Config(), req.ModelIDs, req.Priority); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
        return
    }

    c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func GetModelStatus(c *gin.Context) {
    var req ModelRequest
    if err := c.BindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
        return
    }

    app := c.MustGet("app").(*app.App)
    
    statuses, err := generation.GetModelStatus(c.Request.Context(), app.Config(), req.ModelIDs)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
        return
    }

    c.JSON(http.StatusOK, gin.H{
        "status": "ok",
        "data": statuses,
    })
}