pipeline {
    agent any

    environment {
        PYTHONPATH = "${workspace}"
    }

    stages {
        stage('Setup') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install git+https://github.com/gearlux/logflow.git
                    pip install git+https://github.com/gearlux/confluid.git
                    pip install -e .[test] || pip install -e .
                    pip install pytest pytest-cov flake8 mypy isort black flake8-junit-report
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '''
                    . .venv/bin/activate
                    isort . --check-only
                    black . --check
                    flake8 . --format=junit-xml --output-file=flake8.xml || true
                '''
            }
        }

        stage('Type Check') {
            steps {
                sh '''
                    . .venv/bin/activate
                    mypy .
                '''
            }
        }

        stage('Test') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pytest tests --junitxml=results.xml --cov=dataflux --cov-report=xml --cov-report=term
                '''
            }
        }
    }

    post {
        always {
            junit 'results.xml'
            junit 'flake8.xml'
            recordCoverage(tools: [[parser: 'COBERTURA', pattern: 'coverage.xml']])
        }
        cleanup {
            sh 'rm -rf .venv'
        }
    }
}
