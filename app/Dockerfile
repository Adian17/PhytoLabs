# PhytoLabs web app (Vite → static, served on Railway $PORT)
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Required at Railway build time: https://phytolabs-production.up.railway.app
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

FROM node:20-alpine

RUN npm install -g serve@14.2.4

WORKDIR /app
COPY --from=build /app/dist ./dist

# Railway injects PORT (often 8080). `serve` binds to it.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "serve -s dist -l ${PORT}"]
