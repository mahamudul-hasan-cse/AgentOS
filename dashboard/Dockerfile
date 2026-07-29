# Next.js dashboard.
#
# NEXT_PUBLIC_API_BASE is a BUILD-time argument, not a runtime env var: Next.js
# inlines NEXT_PUBLIC_* into the client bundle at build time. It must therefore
# be the URL the *browser* will use, which is the host-published port
# (http://localhost:8000) — not the compose service name, since the browser
# runs on the host and cannot resolve "backend".

FROM node:20-alpine AS builder

WORKDIR /app

# Dependencies first so source edits do not invalidate the npm layer.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
RUN npm run build

# --- runtime ---------------------------------------------------------------
FROM node:20-alpine

WORKDIR /app
ENV NODE_ENV=production

# Only what `next start` needs.
COPY --from=builder /app/package.json /app/package-lock.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/next.config.mjs ./

EXPOSE 3000

CMD ["npm", "run", "start"]
