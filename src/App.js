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
            href={props.article.DownloadURL.S}
          >
            {props.article.Title.S}
          </Link>
        </Typography>
        <Typography variant="body2" color="textSecondary" component="p">
          {props.article.Abstract.S}
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

      //querying the database by selected label
      const selection_label = this.state.selection.id
      const url = new URL('https://unbqm6hov8.execute-api.us-east-2.amazonaws.com/v1/getarticles')
      const params = { level3: selection_label }
      // assigning label to url
      url.search = new URLSearchParams(params).toString();
      fetch(url)
        .then(res => res.json())
        .then((data) => {
          this.setState({ articlesToDisplay: data.Items })
        })
        .catch(console.log)
    });
  }

  render() {

    const articleCards = this.state.articlesToDisplay.map((article) =>
      <Grid item key={article.SortKey.S}><MediaCard article={article} /></Grid>
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
    articlesToDisplay: []
  };

  componentDidMount() {
    // connect to petal-api to fetch articles list.
    fetch('https://unbqm6hov8.execute-api.us-east-2.amazonaws.com/v1/getalllabels')
    .then(res => res.json())
    .then((data) => {

      let functions = [];
      let labels = data.Items;

      labels.forEach(label => {
        functions.push({
          id: label.Level3.S.toLowerCase().split(' ').join('_'),
          level2: label.Level2.S,
          level3: label.Level3.S
        })
      })

      this.setState({ functions: functions })
    })
    .catch(console.log)
  }

}

export default App;
