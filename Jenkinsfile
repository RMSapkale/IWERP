pipeline {
    agent any

    environment {
        // Base URL & Frontend Configuration
        VITE_API_BASE_URL = "https://iwerp.com/api"
        VITE_ENABLE_ADMIN_DEBUG = "0"
        VITE_ENABLE_PLSQL = "0"

        // Database Configuration
        POSTGRES_USER = 'iwfusion'
        POSTGRES_PASSWORD = 'iwfusion' // Change for production
        POSTGRES_DB = 'iwfusion'

        // Backend & Security Configuration
        SECRET_KEY = '2885bee639ec41b22072e76f2aab1adc' // Change for production
        ALLOWED_ORIGINS = 'https://iwerp.com'
        IWERP_BASE_DIR = '/app/runtime/iwerp-prod'
        
        // AI Model Interface Configuration
        IWFUSION_INFERENCE_BACKEND = 'llama_cpp'
        IWFUSION_MODEL_NAME = 'IWFUSION-SLM-V1'
        IWFUSION_USE_RERANKER = 'false'
        IWFUSION_USE_EMBEDDINGS = 'true'

        // Database Initialization Configuration
        DB_INIT_MODE = 'safe'
        
        // Local Model Performance Configuration
        MODEL_CONTEXT_SIZE = '8192'
        MODEL_THREADS = '8'
        MODEL_GPU_LAYERS = '0'

        // Host Port Mappings to avoid collisions (e.g. Jenkins on 8080)
        PROXY_HTTP_PORT = '80'
        PROXY_HTTPS_PORT = '443'
        BACKEND_PORT = '8000'
        MODEL_PORT = '8090'
        POSTGRES_PORT = '5432'
        REDIS_PORT = '6379'
    }

    stages {
        stage('Cleanup Environment') {
            steps {
                script {
                    echo 'Reclaiming disk space before starting new build...'
                    sh "docker system prune -f"
                    sh "docker image prune -a -f --filter 'until=24h'" 
                }
            }
        }

        stage('Checkout') {
            steps {
                git branch: 'main', 
                    credentialsId: 'GitHub', 
                    url: 'https://github.com/RMSapkale/IWERP.git'
            }
        }

        stage('Deploy') {
            steps {
                script {
                    echo 'Deploying to Production...'
                    sh "docker compose up -d --build --remove-orphans --force-recreate"
                }
            }
        }
        
        stage('Initialize Database') {
            steps {
                script {
                    echo 'Running Database Initializer Profile...'
                    sh "docker compose --profile init up db-bootstrap"
                }
            }
        }
    }

    post {
        always {
            script {
                // Clear all unused data including build cache to free up maximum space
                sh "docker system prune -a -f"
            }
        }
        success {
            echo 'Deployment Pipeline completed successfully!'
        }
        failure {
            echo 'Deployment Pipeline failed. Please check the logs.'
        }
    }
}
