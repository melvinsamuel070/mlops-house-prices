# FROM nginx:alpine

# # Copy custom Nginx config
# COPY default.conf /etc/nginx/conf.d/default.conf

# # Copy templates folder (contains index.html) to Nginx root
# COPY templates/ /usr/share/nginx/html/

# # Copy static files (CSS, JS)
# COPY static/ /usr/share/nginx/html/static/

# EXPOSE 80
# CMD ["nginx", "-g", "daemon off;"]





# Use official Nginx image
FROM nginx:alpine

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom Nginx config
COPY default.conf /etc/nginx/conf.d/

# Copy frontend static files
COPY static /usr/share/nginx/html/static
COPY templates /usr/share/nginx/html

# Expose port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
