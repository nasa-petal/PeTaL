import React, {Component} from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import CardMedia from '@material-ui/core/CardMedia';
import Link from '@material-ui/core/Link';

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

function MediaCard(props) {
  const classes = useStyles();

  return (
    <Card className={classes.root}>
      <CardMedia
        className={classes.media}
        image={props.article.image}
        title={props.article.title}
      />
      <CardContent>
        <Typography gutterBottom variant="h5" component="h2">
          <Link
            color="primary"
            href={props.article.url}
          >
            {props.article.title}
          </Link>
        </Typography>
        <Typography variant="body2" color="textSecondary" component="p">
          {props.article.summary}
        </Typography>
      </CardContent>
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

    const articleCards = this.state.articles.map((article) =>
      <Box my={4}><MediaCard article={article} /></Box>
    );

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
        {articleCards}
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
    ],
    articles: []
  };

  componentDidMount() {
    // connect to locally running petal-api to fetch functions list.
    fetch('http://localhost:8080/v1/functions')
    .then(res => res.json())
    .then((data) => {
      this.setState({ functions: data })
    })
    .catch(console.log)

    // connect to locally running petal-api to fetch wikipedia articles.
    fetch('http://localhost:8080/v1/search')
    .then(res => res.json())
    .then((data) => {
      this.setState({ articles: data })
    })
    .catch(console.log)
  }
}

export default App;
