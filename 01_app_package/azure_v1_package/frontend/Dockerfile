FROM node:20-alpine AS build
WORKDIR /ui

COPY app/frontend/package*.json /ui/
RUN npm ci
COPY app/frontend /ui

ARG VITE_API_BASE_URL
ARG VITE_ENABLE_ADMIN_DEBUG=0
ARG VITE_ENABLE_PLSQL=0
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
ENV VITE_ENABLE_ADMIN_DEBUG=${VITE_ENABLE_ADMIN_DEBUG}
ENV VITE_ENABLE_PLSQL=${VITE_ENABLE_PLSQL}

RUN npm run build

FROM nginx:1.27-alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /ui/dist /usr/share/nginx/html
EXPOSE 80
