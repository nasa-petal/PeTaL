import React, {Component} from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import CardMedia from '@material-ui/core/CardMedia';
import Button from '@material-ui/core/Button';
import lizard from './static/images/contemplative-reptile.jpg';

import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';
import TextField from '@material-ui/core/TextField';
import Autocomplete from '@material-ui/lab/Autocomplete';
import Pagination from '@material-ui/lab/Pagination';

const useStyles = makeStyles({
  media: {
    height: 140,
  },
});

function MediaCard() {
  const classes = useStyles();

  return (
    <Card className={classes.root}>
      <CardMedia
        className={classes.media}
        image={lizard}
        title="Contemplative Reptile"
      />
      <CardContent>
        <Typography gutterBottom variant="h5" component="h2">
          Lizard
        </Typography>
        <Typography variant="body2" color="textSecondary" component="p">
          Lizards are a widespread group of squamate reptiles, with over 6,000 species, ranging
          across all continents except Antarctica
        </Typography>
      </CardContent>
      <CardActions>
        <Button size="small" color="primary">
          Learn More
        </Button>
      </CardActions>
    </Card>
  );
}

class WikipediaArticle extends Component {
  render() {
    return (
      <div></div>
    )
  }
}

class Results extends Component {
  render() {
    return (
      <div></div>
    )
  }
}

class App extends Component {
  render() {
    return (
      <Container maxWidth="md">
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
        <Box my={4}><MediaCard /></Box>
        <Box my={4}><MediaCard /></Box>
        <Box my={4}><MediaCard /></Box>
        <Box my={4}><MediaCard /></Box>
        <Box my={4}><Pagination count={10} color="primary" showFirstButton showLastButton /></Box>
        <Results />
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
    fetch('http://localhost:8080/v1/functions')
      .then(res => res.json())
      .then((data) => {
        this.setState({ functions: data })
      })
      .catch(console.log)
  }
}

export default App;
