pipeline {
    agent any

    environment {
        VENV_PATH = "${WORKSPACE}/.venv"
        VENV_BIN = "${VENV_PATH}/bin"
        FORCE_JAVASCRIPT_ACTIONS_TO_NODE24 = 'true'
    }

    stages {
        stage('Initialize') {
            steps {
                echo 'Creating Isolated Virtual Environment...'
                sh "python3 -m venv ${VENV_PATH}"
                echo 'Installing Dependencies...'
                sh "${VENV_BIN}/pip install --upgrade pip"
                sh "${VENV_BIN}/pip uninstall -y logflow confluid || true"
                sh "${VENV_BIN}/pip install --no-cache-dir git+https://github.com/Gearlux/logflow.git@main"
                sh "${VENV_BIN}/pip install --no-cache-dir git+https://github.com/Gearlux/confluid.git@main"
                sh "${VENV_BIN}/pip install -e .[dev] || ${VENV_BIN}/pip install -e ."
            }
        }

        stage('Quality Gates') {
            parallel {
                stage('Black') {
                    steps {
                        script {
                            def rc = sh(script: "${VENV_BIN}/black --check --diff dataflux tests examples > black-diff.txt 2>&1", returnStatus: true)
                            sh """${VENV_BIN}/python3 -c "
import sys, os
lines = open('black-diff.txt').readlines()
with open('black-checkstyle.xml', 'w') as f:
    f.write('<?xml version=\\"1.0\\" encoding=\\"UTF-8\\"?>\\n<checkstyle version=\\"5.0\\">\\n')
    for line in lines:
        if line.startswith('would reformat '):
            path = line.replace('would reformat ', '').strip()
            f.write('  <file name=\\"' + path + '\\">\\n')
            f.write('    <error line=\\"1\\" severity=\\"warning\\" message=\\"Black would reformat this file\\" source=\\"black\\"/>\\n')
            f.write('  </file>\\n')
    f.write('</checkstyle>\\n')
" """
                        }
                    }
                    post {
                        always {
                            recordIssues(
                                id: 'black-dataflux',
                                name: 'Black Formatting (DataFlux)',
                                tools: [checkStyle(pattern: 'black-checkstyle.xml')]
                            )
                        }
                    }
                }
                stage('Isort') {
                    steps {
                        script {
                            def rc = sh(script: "${VENV_BIN}/isort --check-only --diff dataflux tests examples > isort-diff.txt 2>&1", returnStatus: true)
                            sh """${VENV_BIN}/python3 -c "
import sys, os
lines = open('isort-diff.txt').readlines()
with open('isort-checkstyle.xml', 'w') as f:
    f.write('<?xml version=\\"1.0\\" encoding=\\"UTF-8\\"?>\\n<checkstyle version=\\"5.0\\">\\n')
    for line in lines:
        if line.startswith('ERROR: '):
            path = line.split(' ')[1].strip()
            f.write('  <file name=\\"' + path + '\\">\\n')
            f.write('    <error line=\\"1\\" severity=\\"warning\\" message=\\"Isort import order issues\\" source=\\"isort\\"/>\\n')
            f.write('  </file>\\n')
    f.write('</checkstyle>\\n')
" """
                        }
                    }
                    post {
                        always {
                            recordIssues(
                                id: 'isort-dataflux',
                                name: 'Isort Import Order (DataFlux)',
                                tools: [checkStyle(pattern: 'isort-checkstyle.xml')]
                            )
                        }
                    }
                }
                stage('Flake8') {
                    steps {
                        sh "${VENV_BIN}/flake8 dataflux tests examples --tee --output-file=flake8.txt || true"
                    }
                    post {
                        always {
                            recordIssues(
                                id: 'flake8-dataflux',
                                name: 'Flake8 (DataFlux)',
                                tools: [flake8(pattern: 'flake8.txt')]
                            )
                        }
                    }
                }
                stage('Mypy') {
                    steps {
                        sh "${VENV_BIN}/mypy dataflux tests examples > mypy.txt || true"
                    }
                    post {
                        always {
                            recordIssues(
                                id: 'mypy-dataflux',
                                name: 'Mypy (DataFlux)',
                                tools: [myPy(pattern: 'mypy.txt')]
                            )
                        }
                    }
                }
            }
        }

        stage('Unit Tests') {
            steps {
                sh "${VENV_BIN}/pytest tests --junitxml=test-report.xml --cov=dataflux --cov-report=xml:coverage.xml --cov-report=term"
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-report.xml'
                    recordCoverage(
                        id: 'coverage',
                        name: 'Code Coverage',
                        tools: [[parser: 'COBERTURA', pattern: 'coverage.xml']]
                    )
                }
            }
        }

        stage('Verify Examples') {
            steps {
                echo 'Running project examples...'
                sh '''
                    for f in examples/*.py; do
                        echo "Verifying $f..."
                        ${VENV_BIN}/python3 "$f"
                    done
                '''
            }
        }
    }

    post {
        always {
            echo 'DataFlux Pipeline Complete.'
        }
        success {
            echo 'DataFlux is healthy and ready for publication.'
        }
        failure {
            echo 'DataFlux build failed. Please check linting or test failures.'
        }
    }
}
