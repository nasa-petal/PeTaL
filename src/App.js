import React, {Component} from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import CardMedia from '@material-ui/core/CardMedia';
import Link from '@material-ui/core/Link';
import Grid from '@material-ui/core/Grid';

import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';
import TextField from '@material-ui/core/TextField';
import Autocomplete from '@material-ui/lab/Autocomplete';
import Pagination from '@material-ui/lab/Pagination';

const useStyles = makeStyles({
  root: {
    maxWidth: 345,
    height: '100%'
  },
  media: {
    height: 140,
  },
});

function MediaCard(props) {
  const classes = useStyles();

  return (
    <Card className={classes.root}>
      <CardContent>
        <Typography gutterBottom variant="h5" component="h2">
          <Link
            color="primary"
            href={props.article.downloadUrl}
          >
            {props.article.title}
          </Link>
        </Typography>
        <Typography variant="body2" color="textSecondary" component="p">
          {props.article.description}
        </Typography>
      </CardContent>
    </Card>
  );
}

class Results extends Component {
  render() {
    return (
      <div></div>
    )
  }
}

class App extends Component {

  constructor(props) {
    super(props);
    this.onSelectionChange = this.onSelectionChange.bind(this);
  }

  onSelectionChange = (event, values) => {
    this.setState({
      selection: values
    }, () => {
      //if the selection is X'd out, just fetch original articles
      if (this.state.selection == null) {
        this.setState({ articlesToDisplay: [] })
        return;
      }
      //show articles filtered by selected label

      let coreIds = this.state.articles[this.state.selection.level2][this.state.selection.level3];

      //query core api for details for each coreid in the coreids array.

      fetch('https://core.ac.uk:443/api-v2/articles/get?metadata=true&fulltext=false&citations=false&similar=false&duplicate=false&faithfulMetadata=false&apiKey=0RJ98zruXYjW76EThkvwH1s3CycAoleb', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(coreIds)
      })
      .then(res => res.json())
      .then((data) => {
        this.setState({ articlesToDisplay: data })
      })
      .catch(console.log);

    });
  }

  render() {

    const articleCards = this.state.articlesToDisplay.map((article) =>
      <Grid item key={article.data.id}><MediaCard article={article.data} /></Grid>
    );

    return (
      <Container maxWidth="lg">
        <Box my={4}>
          <Typography variant="h4" component="h1" gutterBottom>
            How does nature...
          </Typography>
          <Autocomplete
            id="function"
            options={this.state.functions.sort((a, b) => -b.level2.localeCompare(a.level2))}
            groupBy={(option) => option.level2}
            getOptionLabel={(option) => option.level3}
            style={{ width: 300 }}
            onChange={this.onSelectionChange}
            renderInput={(params) => <TextField {...params} label="" variant="outlined" />}
          />
        </Box>
        <Grid
          container
          spacing={2}
          direction="row"
          justify="flex-start"
          alignItems="stretch"
        >
        {articleCards}
        </Grid>
        <Results />
      </Container>
    )
  }
  
  state = {
    selection: [],
    functions: [],
    articles: {},
    articlesToDisplay: []
  };

  componentDidMount() {
    // connect to petal-api to fetch articles list.
    fetch('https://unbqm6hov8.execute-api.us-east-2.amazonaws.com/v1/getallarticles')
    .then(res => res.json())
    .then((data) => {

      let articles = {};
      let functions = [];
      let articlesRaw = data.Items;

      articlesRaw.forEach(article => {
        articles[article.Level2.S] = articles[article.Level2.S] || {};
        articles[article.Level2.S][article.Level3.S] = [article.CoreId.S];
        //search the functions array and try to find an object with level3 and level2 keys with values that match this article's level2 and level3 values. If one does not exist add a new object to the array.

        let funcExists = functions.find(item => {
          if (item.level2 == article.Level2.S && item.level3 == article.Level3.S) {
            return true;
          }
          return false;
        });

        if(!funcExists) {
          functions.push({
            level2: article.Level2.S,
            level3: article.Level3.S,
          });
        }
      });

      this.setState({ functions: functions })
      this.setState({ articles: articles })
    })
    .catch(console.log)
  }

}

export default App;
