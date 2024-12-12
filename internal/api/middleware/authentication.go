package middleware

import (
	"database/sql"
	"errors"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/utils/hashutil"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

func AuthenticationMiddleware(ctx *gin.Context) {
	authorization := ctx.Request.Header.Get("Authorization")
	apikey := ctx.Request.Header.Get("X-API-Key")

	app := ctx.MustGet("app").(*app.App)

	if apikey != "" {
		apikeyHash := hashutil.Sha3256Hash([]byte(apikey))
		result, err := app.APIKeyRepository.GetAPIKeyWithHash(ctx.Request.Context(), apikeyHash)
        if err != nil {
            if errors.Is(err, sql.ErrNoRows) {
                ctx.JSON(401, gin.H{"message": "The provided API key is invalid"})
                ctx.Abort()
				return
			}

			// Database error
			app.Logger.Error("Database error while checking API key", zap.Error(err))
			ctx.JSON(500, gin.H{"message": "Internal server error checking api-keys in database"})
			ctx.Abort()
			return
        }

		if result.IsRevoked {
			ctx.JSON(401, gin.H{"message": "The provided API key is revoked"})
			ctx.Abort()
			return
		}
	} else if authorization != "" {
		// TODO: implement token based authorization
		ctx.JSON(401, gin.H{"message": "Token based authorization is not allowed"})
		ctx.Abort()
		return
	} else {
		ctx.JSON(401, gin.H{"message": "Unauthorized access"})
		ctx.Abort()
		return
	}

	ctx.Next()
}
