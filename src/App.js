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
    //maxWidth: 345,
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
            target="_blank"
            rel="noopener noreferrer"
            href={props.article.S ? props.article.S[0].U : ''}
          >
            {props.article.DN}
          </Link>
        </Typography>
        <Typography variant="body2" color="textSecondary" component="p">
          MAG topics:
        {props.article.F.map((topic, index) => (
          <span>{ " " + topic.FN + ',' }</span>
        ))}
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

      // Query MAG

      
      switch(this.state.selection.id) {
        case 'modify/convert_thermal_energy':
          query = "Or(Composite(F.FN=='thermal converter'),Composite(F.FN=='thermal control'),Composite(F.FN=='thermal balance'))";
          break;
        case 'distribute_energy':
          query = "Or(Composite(F.FN=='thermal distribution'),Composite(F.FN=='heat transfer process'),Composite(F.FN=='transient heat transfer'),Composite(F.FN=='heat spreading'),Composite(F.FN=='heat flow'),Composite(F.FN=='heat transfer fluid'),Composite(F.FN=='thermal emission'),Composite(F.FN=='thermal transport'),Composite(F.FN=='bioheat transfer'),Composite(F.FN=='heat transfer model'),Composite(F.FN=='heat spreading'))";
          break;
        case 'protect_from_temperature':
          query = "Or(Composite(F.FN=='thermal resistance'),Composite(F.FN=='passive cooling'),Composite(F.FN=='thermal control'),Composite(F.FN=='thermal fatigue'),Composite(F.FN=='thermal strain'),Composite(F.FN=='thermal dissipation'),Composite(F.FN=='convective cooling'),Composite(F.FN=='thermal buckling'),Composite(F.FN=='thermal residual stress'),Composite(F.FN=='thermal degradation of polymers'),Composite(F.FN=='space shuttle thermal protection system'),Composite(F.FN=='heat spreading'),Composite(F.FN=='heat tolerance'),Composite(F.FN=='thermoregulation'))";
          // ,Composite(F.FN=='temperature stress'),Composite(F.FN=='heat stress')
          break;
        case 'sense_temperature_cues':
          query = "Or(Composite(F.FN=='thermal sensing'),Composite(F.FN=='temperature measurement'),Composite(F.FN=='temperature monitoring'),Composite(F.FN=='temperature sensing'),Composite(F.FN=='thermal detector'),Composite(F.FN=='thermal monitoring'),Composite(F.FN=='thermal probe'),Composite(F.FN=='thermal sensors'),Composite(F.FN=='infrared thermal imaging'))";
          break;
        case 'store_energy':
          query = "Composite(F.FN=='solar thermal collector')";
          break;
      }

      var query = "And(Ty='0'," + query + ",Or(Composite(J.JN=='biomimetics'), Composite(F.FN=='biology')))";

      const url = new URL('https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate');
      const params = {
        expr: query,
        model: 'latest',
        count: 20000,
        offset: 0,
        attributes: 'Id,DOI,DN,VFN,F.FN,AA.AuId,AW,RId,S'
      }
      url.search = new URLSearchParams(params).toString();

      fetch(url, {
        headers: {
          'Ocp-Apim-Subscription-Key': 'd969a6c4bef34765ae7c5f0e75dd624e'
        }
      })
        .then(res => res.json())
        .then((data) => {
          console.log(data);
          this.setState({ articlesToDisplay: data.entities })
        })
        .catch(console.log)

      //querying the database by selected label
      /*const selection_label = this.state.selection.id
      const url = new URL('https://unbqm6hov8.execute-api.us-east-2.amazonaws.com/v1/getarticles')
      const params = { level3: selection_label }
      // assigning label to url
      url.search = new URLSearchParams(params).toString();
      fetch(url)
        .then(res => res.json())
        .then((data) => {
          this.setState({ articlesToDisplay: data.Items })
        })
        .catch(console.log)*/
    });
  }

  render() {

    const articleCards = this.state.articlesToDisplay.map((article) =>
      <Grid item xs={12} key={article.Id}><MediaCard article={article} /></Grid>
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
        if(['Modify/convert thermal energy', 'Sense temperature cues', 'Protect from temperature', 'Store energy', 'Distribute energy'].includes(label.Level3.S)){
          functions.push({
            id: label.Level3.S.toLowerCase().split(' ').join('_'),
            level2: label.Level2.S,
            level3: label.Level3.S
          })
        }
      })

      this.setState({ functions: functions })
    })
    .catch(console.log)
  }

}

export default App;
