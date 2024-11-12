We've given up on using the OpenAPI generator to generate clients for this project; it's just impossible to get the client to work with streaming or with message pack. The client-code needs to be hand-written rather than auto-generated.

Anyway, if you want auto-generated clients, use the OpenAPI generator like this:

```bash
yarn global add @openapitools/openapi-generator-cli
```

Typescript Fetch Client:
```bash
openapi-generator-cli generate -i ./v1/openapi.yaml -g typescript-fetch -o ./client-ts
```

Go Client:
```bash
openapi-generator-cli generate -i ./v1/openapi.yaml -g go -o ./client-go
```


