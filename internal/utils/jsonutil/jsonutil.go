package jsonutil

import "encoding/json"

func MapToStruct(source map[string]any, target interface{}) error {
	data, err := json.Marshal(source)
	if err != nil {
		return err
	}

	err = json.Unmarshal(data, target)
	if err != nil {
		return err
	}

	return nil
}

func StructToMap(source interface{}) (map[string]any, error) {
	data, err := json.Marshal(source)
	if err != nil {
		return nil, err
	}

	var target map[string]any
	err = json.Unmarshal(data, &target)
	if err != nil {
		return nil, err
	}

	return target, nil
}
