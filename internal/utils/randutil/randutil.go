package randutil

import (
	"crypto/rand"
	"encoding/base64"
	"strings"
)

func RandomString(length int) (string, error) {
	key := make([]byte, length)

	if _, err := rand.Read(key); err != nil {
		return "", err
	}

	return base64.RawURLEncoding.EncodeToString(key), nil
}

func MaskString(apiKey string, visibleStart, visibleEnd int) string {
	if len(apiKey) <= visibleStart+visibleEnd {
		return apiKey
	}

	start := apiKey[:visibleStart]
	end := apiKey[len(apiKey)-visibleEnd:]
	masked := start + strings.Repeat("*", len(apiKey)-(visibleStart+visibleEnd)) + end
	return masked
}
