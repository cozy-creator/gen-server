package utils

import "lukechampine.com/blake3"

func Blake3Hash(data []byte) [32]byte {
	return blake3.Sum256(data)
}
