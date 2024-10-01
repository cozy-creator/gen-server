package hashutil

import (
	"encoding/hex"

	"golang.org/x/crypto/sha3"
	"lukechampine.com/blake3"
)

func Blake3Hash(data []byte) string {
	hash := blake3.Sum256(data)
	return hex.EncodeToString(hash[:])
}

func Sha3256Hash(data []byte) string {
	hash := sha3.Sum256(data)
	return hex.EncodeToString(hash[:])
}
