import React, {Component} from 'react';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';
import TextField from '@material-ui/core/TextField';
import Autocomplete from '@material-ui/lab/Autocomplete';

class App extends Component {
    render() {
        return (
          <Container maxWidth="sm">
            <Box my={4}>
              <Typography variant="h4" component="h1" gutterBottom>
                How does nature...
              </Typography>
              <Autocomplete
                id="function"
                options={this.state.functions}
                getOptionLabel={(option) => option.label}
                style={{ width: 300 }}
                renderInput={(params) => <TextField {...params} label="" variant="outlined" />}
              />
            </Box>
          </Container>
        )
    }

    state = {
        functions: [
          { label: 'Reduce drag', id: 1 },
          { label: 'Absorb shock', id: 2 },
          { label: 'Dissipate heat', id: 3 },
          { label: 'Increase lift', id: 4 },
          { label: 'Remove particles from a surface', id: 5 }
        ]
    };

    componentDidMount() {
        // connect to locally running petal-api to fetch functions list.
        fetch('http://localhost:8080')
            .then(res => res.json())
            .then((data) => {
                this.setState({ functions: data })
            })
            .catch(console.log)
    }
}

export default App;
