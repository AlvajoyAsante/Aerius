# Contributing to Aerius

Thank you for your interest in contributing to Aerius! This document provides guidelines and information for contributors.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Maintain professional communication

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported
2. Create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Node version, browser)
   - Screenshots if applicable

### Suggesting Features

1. Check existing feature requests
2. Create a new issue describing:
   - Use case and problem it solves
   - Proposed solution
   - Alternative approaches considered
   - Mockups or examples if applicable

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Write/update tests
5. Update documentation
6. Commit with clear messages
7. Push to your fork
8. Create a Pull Request

## Development Setup

### Prerequisites
- Node.js v14 or higher
- MongoDB
- Git

### Initial Setup
```bash
git clone https://github.com/AlvajoyAsante/Aerius.git
cd Aerius

# Backend
cd backend
npm install
cp .env.example .env
# Configure .env

# Frontend
cd ../frontend
npm install
```

### Running Locally
```bash
# Terminal 1 - Backend
cd backend
npm run dev

# Terminal 2 - Frontend
cd frontend
npm start

# Terminal 3 - MongoDB (if local)
mongod
```

## Coding Standards

### JavaScript Style
- Use ES6+ features
- Use async/await for asynchronous code
- Use meaningful variable names
- Keep functions small and focused
- Add comments for complex logic

### React Components
- Use functional components with hooks
- Keep components small and reusable
- Use PropTypes or TypeScript for type checking
- Follow React best practices

### Backend API
- Follow RESTful conventions
- Validate input data
- Handle errors gracefully
- Return appropriate HTTP status codes
- Document API endpoints

### Git Commits
Format: `type: description`

Types:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Examples:
```
feat: add export survey data feature
fix: resolve drone status update issue
docs: update API documentation
```

## Testing

### Backend Tests
```bash
cd backend
npm test
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Test Coverage
- Aim for >80% code coverage
- Test critical paths thoroughly
- Include edge cases

## Documentation

### Code Documentation
- Document complex functions
- Explain non-obvious logic
- Keep comments up-to-date

### API Documentation
- Document all endpoints
- Include request/response examples
- Note authentication requirements
- List possible error responses

### User Documentation
- Update README for major changes
- Add examples for new features
- Keep installation guide current

## Project Structure

```
Aerius/
├── backend/
│   ├── models/           # Database models
│   ├── routes/           # API routes
│   ├── controllers/      # Business logic
│   ├── middleware/       # Custom middleware
│   ├── utils/            # Helper functions
│   └── server.js         # Express app
│
├── frontend/
│   └── src/
│       ├── components/   # Reusable components
│       ├── pages/        # Page components
│       ├── services/     # API services
│       ├── utils/        # Helper functions
│       └── App.js        # Main app
│
└── docs/                 # Documentation
```

## Adding New Features

### Backend Feature
1. Create/update model in `models/`
2. Add controller in `controllers/`
3. Create routes in `routes/`
4. Update `server.js` if needed
5. Add tests
6. Document API endpoint

### Frontend Feature
1. Create component in `components/` or `pages/`
2. Add API calls in `services/`
3. Update routing if needed
4. Add styling
5. Add tests
6. Update documentation

## Release Process

1. Update version in `package.json`
2. Update CHANGELOG.md
3. Create release branch
4. Run all tests
5. Create pull request
6. Merge to main after review
7. Tag release
8. Deploy to production

## Questions?

- Create an issue for questions
- Join discussions in existing issues
- Contact maintainers

## License

By contributing, you agree that your contributions will be licensed under the ISC License.
