package main

import (
	"cozy-creator/go-cozy/cmd"
	"fmt"
	"os"
)

func main() {
	rootCmd := cmd.GetRootCmd()
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
