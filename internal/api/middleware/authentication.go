package middleware

import (
	"github.com/gin-gonic/gin"
)

func AuthenticationMiddleware(c *gin.Context) {
	// authorization := c.Request.Header.Get("Authorization")
	// apikey := c.Request.Header.Get("X-API-Key")

	// app := c.MustGet("app").(*app.App)

	// if apikey != "" {
	// 	apikey := hashutil.Sha3256Hash([]byte(apikey))
	// 	result, err := app.DB().GetAPIKey(c.Request.Context(), apikey)
	// 	if err != nil {
	// 		app.Logger.Error("Failed to get API key", zap.Error(err))
	// 		c.JSON(401, gin.H{"message": "The provided API key is invalid"})
	// 		c.Abort()
	// 		return
	// 	}

	// 	if result.IsRevoked {
	// 		c.JSON(401, gin.H{"message": "The provided API key is revoked"})
	// 		c.Abort()
	// 		return
	// 	}
	// } else if authorization != "" {
	// 	// TODO: implement token based authorization
	// 	c.JSON(401, gin.H{"message": "Token based authorization is not allowed"})
	// 	c.Abort()
	// 	return
	// } else {
	// 	c.JSON(401, gin.H{"message": "Unauthorized access"})
	// 	c.Abort()
	// 	return
	// }

	c.Next()
}
