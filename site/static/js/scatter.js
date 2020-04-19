// Reference
// https://rawgit.com/domitry/elegans/master/release/elegans.js
// http://bl.ocks.org/domitry/11322575

(function (root, initialize){
    var Elegans = initialize();
    if(typeof define !== "undefined" && define.amd)define(Elegans);
    root.Elegans = Elegans;
}(this, function(){
    //modules here
/**
 * @license almond 0.2.9 Copyright (c) 2011-2014, The Dojo Foundation All Rights Reserved.
 * Available via the MIT or new BSD license.
 * see: http://github.com/jrburke/almond for details
 */
//Going sloppy to avoid 'use strict' string cost, but strict practices should
//be followed.
/*jslint sloppy: true */
/*global setTimeout: false */

var requirejs, require, define;
(function (undef) {
    var main, req, makeMap, handlers,
        defined = {},
        waiting = {},
        config = {},
        defining = {},
        hasOwn = Object.prototype.hasOwnProperty,
        aps = [].slice,
        jsSuffixRegExp = /\.js$/;

    function hasProp(obj, prop) {
        return hasOwn.call(obj, prop);
    }

    /**
     * Given a relative module name, like ./something, normalize it to
     * a real name that can be mapped to a path.
     * @param {String} name the relative name
     * @param {String} baseName a real name that the name arg is relative
     * to.
     * @returns {String} normalized name
     */
    function normalize(name, baseName) {
        var nameParts, nameSegment, mapValue, foundMap, lastIndex,
            foundI, foundStarMap, starI, i, j, part,
            baseParts = baseName && baseName.split("/"),
            map = config.map,
            starMap = (map && map['*']) || {};

        //Adjust any relative paths.
        if (name && name.charAt(0) === ".") {
            //If have a base name, try to normalize against it,
            //otherwise, assume it is a top-level require that will
            //be relative to baseUrl in the end.
            if (baseName) {
                //Convert baseName to array, and lop off the last part,
                //so that . matches that "directory" and not name of the baseName's
                //module. For instance, baseName of "one/two/three", maps to
                //"one/two/three.js", but we want the directory, "one/two" for
                //this normalization.
                baseParts = baseParts.slice(0, baseParts.length - 1);
                name = name.split('/');
                lastIndex = name.length - 1;

                // Node .js allowance:
                if (config.nodeIdCompat && jsSuffixRegExp.test(name[lastIndex])) {
                    name[lastIndex] = name[lastIndex].replace(jsSuffixRegExp, '');
                }

                name = baseParts.concat(name);

                //start trimDots
                for (i = 0; i < name.length; i += 1) {
                    part = name[i];
                    if (part === ".") {
                        name.splice(i, 1);
                        i -= 1;
                    } else if (part === "..") {
                        if (i === 1 && (name[2] === '..' || name[0] === '..')) {
                            //End of the line. Keep at least one non-dot
                            //path segment at the front so it can be mapped
                            //correctly to disk. Otherwise, there is likely
                            //no path mapping for a path starting with '..'.
                            //This can still fail, but catches the most reasonable
                            //uses of ..
                            break;
                        } else if (i > 0) {
                            name.splice(i - 1, 2);
                            i -= 2;
                        }
                    }
                }
                //end trimDots

                name = name.join("/");
            } else if (name.indexOf('./') === 0) {
                // No baseName, so this is ID is resolved relative
                // to baseUrl, pull off the leading dot.
                name = name.substring(2);
            }
        }

        //Apply map config if available.
        if ((baseParts || starMap) && map) {
            nameParts = name.split('/');

            for (i = nameParts.length; i > 0; i -= 1) {
                nameSegment = nameParts.slice(0, i).join("/");

                if (baseParts) {
                    //Find the longest baseName segment match in the config.
                    //So, do joins on the biggest to smallest lengths of baseParts.
                    for (j = baseParts.length; j > 0; j -= 1) {
                        mapValue = map[baseParts.slice(0, j).join('/')];

                        //baseName segment has  config, find if it has one for
                        //this name.
                        if (mapValue) {
                            mapValue = mapValue[nameSegment];
                            if (mapValue) {
                                //Match, update name to the new value.
                                foundMap = mapValue;
                                foundI = i;
                                break;
                            }
                        }
                    }
                }

                if (foundMap) {
                    break;
                }

                //Check for a star map match, but just hold on to it,
                //if there is a shorter segment match later in a matching
                //config, then favor over this star map.
                if (!foundStarMap && starMap && starMap[nameSegment]) {
                    foundStarMap = starMap[nameSegment];
                    starI = i;
                }
            }

            if (!foundMap && foundStarMap) {
                foundMap = foundStarMap;
                foundI = starI;
            }

            if (foundMap) {
                nameParts.splice(0, foundI, foundMap);
                name = nameParts.join('/');
            }
        }

        return name;
    }

    function makeRequire(relName, forceSync) {
        return function () {
            //A version of a require function that passes a moduleName
            //value for items that may need to
            //look up paths relative to the moduleName
            return req.apply(undef, aps.call(arguments, 0).concat([relName, forceSync]));
        };
    }

    function makeNormalize(relName) {
        return function (name) {
            return normalize(name, relName);
        };
    }

    function makeLoad(depName) {
        return function (value) {
            defined[depName] = value;
        };
    }

    function callDep(name) {
        if (hasProp(waiting, name)) {
            var args = waiting[name];
            delete waiting[name];
            defining[name] = true;
            main.apply(undef, args);
        }

        if (!hasProp(defined, name) && !hasProp(defining, name)) {
            throw new Error('No ' + name);
        }
        return defined[name];
    }

    //Turns a plugin!resource to [plugin, resource]
    //with the plugin being undefined if the name
    //did not have a plugin prefix.
    function splitPrefix(name) {
        var prefix,
            index = name ? name.indexOf('!') : -1;
        if (index > -1) {
            prefix = name.substring(0, index);
            name = name.substring(index + 1, name.length);
        }
        return [prefix, name];
    }

    /**
     * Makes a name map, normalizing the name, and using a plugin
     * for normalization if necessary. Grabs a ref to plugin
     * too, as an optimization.
     */
    makeMap = function (name, relName) {
        var plugin,
            parts = splitPrefix(name),
            prefix = parts[0];

        name = parts[1];

        if (prefix) {
            prefix = normalize(prefix, relName);
            plugin = callDep(prefix);
        }

        //Normalize according
        if (prefix) {
            if (plugin && plugin.normalize) {
                name = plugin.normalize(name, makeNormalize(relName));
            } else {
                name = normalize(name, relName);
            }
        } else {
            name = normalize(name, relName);
            parts = splitPrefix(name);
            prefix = parts[0];
            name = parts[1];
            if (prefix) {
                plugin = callDep(prefix);
            }
        }

        //Using ridiculous property names for space reasons
        return {
            f: prefix ? prefix + '!' + name : name, //fullName
            n: name,
            pr: prefix,
            p: plugin
        };
    };

    function makeConfig(name) {
        return function () {
            return (config && config.config && config.config[name]) || {};
        };
    }

    handlers = {
        require: function (name) {
            return makeRequire(name);
        },
        exports: function (name) {
            var e = defined[name];
            if (typeof e !== 'undefined') {
                return e;
            } else {
                return (defined[name] = {});
            }
        },
        module: function (name) {
            return {
                id: name,
                uri: '',
                exports: defined[name],
                config: makeConfig(name)
            };
        }
    };

    main = function (name, deps, callback, relName) {
        var cjsModule, depName, ret, map, i,
            args = [],
            callbackType = typeof callback,
            usingExports;

        //Use name if no relName
        relName = relName || name;

        //Call the callback to define the module, if necessary.
        if (callbackType === 'undefined' || callbackType === 'function') {
            //Pull out the defined dependencies and pass the ordered
            //values to the callback.
            //Default to [require, exports, module] if no deps
            deps = !deps.length && callback.length ? ['require', 'exports', 'module'] : deps;
            for (i = 0; i < deps.length; i += 1) {
                map = makeMap(deps[i], relName);
                depName = map.f;

                //Fast path CommonJS standard dependencies.
                if (depName === "require") {
                    args[i] = handlers.require(name);
                } else if (depName === "exports") {
                    //CommonJS module spec 1.1
                    args[i] = handlers.exports(name);
                    usingExports = true;
                } else if (depName === "module") {
                    //CommonJS module spec 1.1
                    cjsModule = args[i] = handlers.module(name);
                } else if (hasProp(defined, depName) ||
                           hasProp(waiting, depName) ||
                           hasProp(defining, depName)) {
                    args[i] = callDep(depName);
                } else if (map.p) {
                    map.p.load(map.n, makeRequire(relName, true), makeLoad(depName), {});
                    args[i] = defined[depName];
                } else {
                    throw new Error(name + ' missing ' + depName);
                }
            }

            ret = callback ? callback.apply(defined[name], args) : undefined;

            if (name) {
                //If setting exports via "module" is in play,
                //favor that over return value and exports. After that,
                //favor a non-undefined return value over exports use.
                if (cjsModule && cjsModule.exports !== undef &&
                        cjsModule.exports !== defined[name]) {
                    defined[name] = cjsModule.exports;
                } else if (ret !== undef || !usingExports) {
                    //Use the return value from the function.
                    defined[name] = ret;
                }
            }
        } else if (name) {
            //May just be an object definition for the module. Only
            //worry about defining if have a module name.
            defined[name] = callback;
        }
    };

    requirejs = require = req = function (deps, callback, relName, forceSync, alt) {
        if (typeof deps === "string") {
            if (handlers[deps]) {
                //callback in this case is really relName
                return handlers[deps](callback);
            }
            //Just return the module wanted. In this scenario, the
            //deps arg is the module name, and second arg (if passed)
            //is just the relName.
            //Normalize module name, if it contains . or ..
            return callDep(makeMap(deps, callback).f);
        } else if (!deps.splice) {
            //deps is a config object, not an array.
            config = deps;
            if (config.deps) {
                req(config.deps, config.callback);
            }
            if (!callback) {
                return;
            }

            if (callback.splice) {
                //callback is an array, which means it is a dependency list.
                //Adjust args if there are dependencies
                deps = callback;
                callback = relName;
                relName = null;
            } else {
                deps = undef;
            }
        }

        //Support require(['a'])
        callback = callback || function () {};

        //If relName is a function, it is an errback handler,
        //so remove it.
        if (typeof relName === 'function') {
            relName = forceSync;
            forceSync = alt;
        }

        //Simulate async callback;
        if (forceSync) {
            main(undef, deps, callback, relName);
        } else {
            //Using a non-zero value because of concern for what old browsers
            //do, and latest browsers "upgrade" to 4 if lower value is used:
            //http://www.whatwg.org/specs/web-apps/current-work/multipage/timers.html#dom-windowtimers-settimeout:
            //If want a value immediately, use require('id') instead -- something
            //that works in almond on the global level, but not guaranteed and
            //unlikely to work in other AMD implementations.
            setTimeout(function () {
                main(undef, deps, callback, relName);
            }, 4);
        }

        return req;
    };

    /**
     * Just drops the config on the floor, but returns req in case
     * the config return value is used.
     */
    req.config = function (cfg) {
        return req(cfg);
    };

    /**
     * Expose module registry for debugging and tooling
     */
    requirejs._defined = defined;

    define = function (name, deps, callback) {

        //This module may not have dependencies
        if (!deps.splice) {
            //deps is not an array, so probably means
            //an object literal or factory function for
            //the value. Adjust args.
            callback = deps;
            deps = [];
        }

        if (!hasProp(defined, name) && !hasProp(waiting, name)) {
            waiting[name] = [name, deps, callback];
        }
    };

    define.amd = {
        jQuery: true
    };
}());

define("../lib/almond/almond", function(){});

define('utils/TrackballControls',[],function(){
/**
 * @author Eberhard Graether / http://egraether.com/
 * @author Mark Lundin 	/ http://mark-lundin.com
 */

    var TrackballControls = function ( object, domElement ) {

	var _this = this;
	var STATE = { NONE: -1, ROTATE: 0, ZOOM: 1, PAN: 2, TOUCH_ROTATE: 3, TOUCH_ZOOM_PAN: 4 };

	this.object = object;
	this.domElement = ( domElement !== undefined ) ? domElement : document;

	// API

	this.enabled = true;

	this.screen = { left: 0, top: 0, width: 0, height: 0 };

	this.rotateSpeed = 1.0;
	this.zoomSpeed = 1.2;
	this.panSpeed = 0.3;

	this.noRotate = false;
	this.noZoom = false;
	this.noPan = false;
	this.noRoll = false;

	this.staticMoving = false;
	this.dynamicDampingFactor = 0.2;

	this.minDistance = 0;
	this.maxDistance = Infinity;

	this.keys = [ 65 /*A*/, 83 /*S*/, 68 /*D*/ ];

	// internals

	this.target = new THREE.Vector3();

	var EPS = 0.000001;

	var lastPosition = new THREE.Vector3();

	var _state = STATE.NONE,
	    _prevState = STATE.NONE,

	    _eye = new THREE.Vector3(),

	    _rotateStart = new THREE.Vector3(),
	    _rotateEnd = new THREE.Vector3(),

	    _zoomStart = new THREE.Vector2(),
	    _zoomEnd = new THREE.Vector2(),

	    _touchZoomDistanceStart = 0,
	    _touchZoomDistanceEnd = 0,

	    _panStart = new THREE.Vector2(),
	    _panEnd = new THREE.Vector2();

	// for reset

	this.target0 = this.target.clone();
	this.position0 = this.object.position.clone();
	this.up0 = this.object.up.clone();

	// events

	var changeEvent = { type: 'change' };
	var startEvent = { type: 'start'};
	var endEvent = { type: 'end'};


	// methods

	this.handleResize = function () {

	    if ( this.domElement === document ) {

		this.screen.left = 0;
		this.screen.top = 0;
		this.screen.width = window.innerWidth;
		this.screen.height = window.innerHeight;

	    } else {

		var box = this.domElement.getBoundingClientRect();
		// adjustments come from similar code in the jquery offset() function
		var d = this.domElement.ownerDocument.documentElement;
		this.screen.left = box.left + window.pageXOffset - d.clientLeft;
		this.screen.top = box.top + window.pageYOffset - d.clientTop;
		this.screen.width = box.width;
		this.screen.height = box.height;

	    }

	};

	this.handleEvent = function ( event ) {

	    if ( typeof this[ event.type ] == 'function' ) {

		this[ event.type ]( event );

	    }

	};

	var getMouseOnScreen = ( function () {

	    var vector = new THREE.Vector2();

	    return function ( pageX, pageY ) {

		vector.set(
		    ( pageX - _this.screen.left ) / _this.screen.width,
		    ( pageY - _this.screen.top ) / _this.screen.height
		);

		return vector;

	    };

	}() );

	var getMouseProjectionOnBall = ( function () {

	    var vector = new THREE.Vector3();
	    var objectUp = new THREE.Vector3();
	    var mouseOnBall = new THREE.Vector3();

	    return function ( pageX, pageY ) {

		mouseOnBall.set(
		    ( pageX - _this.screen.width * 0.5 - _this.screen.left ) / (_this.screen.width*.5),
		    ( _this.screen.height * 0.5 + _this.screen.top - pageY ) / (_this.screen.height*.5),
		    0.0
		);

		var length = mouseOnBall.length();

		if ( _this.noRoll ) {

		    if ( length < Math.SQRT1_2 ) {

			mouseOnBall.z = Math.sqrt( 1.0 - length*length );

		    } else {

			mouseOnBall.z = .5 / length;
			
		    }

		} else if ( length > 1.0 ) {

		    mouseOnBall.normalize();

		} else {

		    mouseOnBall.z = Math.sqrt( 1.0 - length * length );

		}

		_eye.copy( _this.object.position ).sub( _this.target );

		vector.copy( _this.object.up ).setLength( mouseOnBall.y );
		vector.add( objectUp.copy( _this.object.up ).cross( _eye ).setLength( mouseOnBall.x ) );
		vector.add( _eye.setLength( mouseOnBall.z ) );

		return vector;

	    };

	}() );

	this.rotateCamera = (function(){

	    var axis = new THREE.Vector3(),
		quaternion = new THREE.Quaternion();


	    return function () {
		var angle = Math.acos( _rotateStart.dot( _rotateEnd ) / _rotateStart.length() / _rotateEnd.length() );

		if ( angle ) {

		    axis.crossVectors( _rotateStart, _rotateEnd ).normalize();

		    angle *= _this.rotateSpeed;

		    quaternion.setFromAxisAngle( axis, -angle );

		    _eye.applyQuaternion( quaternion );
		    _this.object.up.applyQuaternion( quaternion );

		    _rotateEnd.applyQuaternion( quaternion );

		    if ( _this.staticMoving ) {

			_rotateStart.copy( _rotateEnd );

		    } else {

			quaternion.setFromAxisAngle( axis, angle * ( _this.dynamicDampingFactor - 1.0 ) );
			_rotateStart.applyQuaternion( quaternion );

		    }

		}
	    };

	}());

	this.zoomCamera = function () {

	    if ( _state === STATE.TOUCH_ZOOM_PAN ) {

		var factor = _touchZoomDistanceStart / _touchZoomDistanceEnd;
		_touchZoomDistanceStart = _touchZoomDistanceEnd;
		_eye.multiplyScalar( factor );

	    } else {

		var factor = 1.0 + ( _zoomEnd.y - _zoomStart.y ) * _this.zoomSpeed;

		if ( factor !== 1.0 && factor > 0.0 ) {

		    _eye.multiplyScalar( factor );

		    if ( _this.staticMoving ) {

			_zoomStart.copy( _zoomEnd );

		    } else {

			_zoomStart.y += ( _zoomEnd.y - _zoomStart.y ) * this.dynamicDampingFactor;

		    }

		}

	    }

	};

	this.panCamera = (function(){

	    var mouseChange = new THREE.Vector2(),
		objectUp = new THREE.Vector3(),
		pan = new THREE.Vector3();

	    return function () {

		mouseChange.copy( _panEnd ).sub( _panStart );

		if ( mouseChange.lengthSq() ) {

		    mouseChange.multiplyScalar( _eye.length() * _this.panSpeed );

		    pan.copy( _eye ).cross( _this.object.up ).setLength( mouseChange.x );
		    pan.add( objectUp.copy( _this.object.up ).setLength( mouseChange.y ) );

		    _this.object.position.add( pan );
		    _this.target.add( pan );

		    if ( _this.staticMoving ) {

			_panStart.copy( _panEnd );

		    } else {

			_panStart.add( mouseChange.subVectors( _panEnd, _panStart ).multiplyScalar( _this.dynamicDampingFactor ) );

		    }

		}
	    };

	}());

	this.checkDistances = function () {

	    if ( !_this.noZoom || !_this.noPan ) {

		if ( _eye.lengthSq() > _this.maxDistance * _this.maxDistance ) {

		    _this.object.position.addVectors( _this.target, _eye.setLength( _this.maxDistance ) );

		}

		if ( _eye.lengthSq() < _this.minDistance * _this.minDistance ) {

		    _this.object.position.addVectors( _this.target, _eye.setLength( _this.minDistance ) );

		}

	    }

	};

	this.update = function () {

	    _eye.subVectors( _this.object.position, _this.target );

	    if ( !_this.noRotate ) {

		_this.rotateCamera();

	    }

	    if ( !_this.noZoom ) {

		_this.zoomCamera();

	    }

	    if ( !_this.noPan ) {

		_this.panCamera();

	    }

	    _this.object.position.addVectors( _this.target, _eye );

	    _this.checkDistances();

	    _this.object.lookAt( _this.target );

	    if ( lastPosition.distanceToSquared( _this.object.position ) > EPS ) {

		_this.dispatchEvent( changeEvent );

		lastPosition.copy( _this.object.position );

	    }

	};

	this.reset = function () {

	    _state = STATE.NONE;
	    _prevState = STATE.NONE;

	    _this.target.copy( _this.target0 );
	    _this.object.position.copy( _this.position0 );
	    _this.object.up.copy( _this.up0 );

	    _eye.subVectors( _this.object.position, _this.target );

	    _this.object.lookAt( _this.target );

	    _this.dispatchEvent( changeEvent );

	    lastPosition.copy( _this.object.position );

	};

	// listeners

	function keydown( event ) {

	    if ( _this.enabled === false ) return;

	    window.removeEventListener( 'keydown', keydown );

	    _prevState = _state;

	    if ( _state !== STATE.NONE ) {

		return;

	    } else if ( event.keyCode === _this.keys[ STATE.ROTATE ] && !_this.noRotate ) {

		_state = STATE.ROTATE;

	    } else if ( event.keyCode === _this.keys[ STATE.ZOOM ] && !_this.noZoom ) {

		_state = STATE.ZOOM;

	    } else if ( event.keyCode === _this.keys[ STATE.PAN ] && !_this.noPan ) {

		_state = STATE.PAN;

	    }

	}

	function keyup( event ) {

	    if ( _this.enabled === false ) return;

	    _state = _prevState;

	    window.addEventListener( 'keydown', keydown, false );

	}

	function mousedown( event ) {

	    if ( _this.enabled === false ) return;

	    event.preventDefault();
	    event.stopPropagation();

	    if ( _state === STATE.NONE ) {

		_state = event.button;

	    }

	    if ( _state === STATE.ROTATE && !_this.noRotate ) {
		_rotateStart.copy(getMouseProjectionOnBall( event.layerX, event.layerY ));
		_rotateEnd.copy( _rotateStart );

	    } else if ( _state === STATE.ZOOM && !_this.noZoom ) {

		_zoomStart.copy( getMouseOnScreen( event.layerX, event.layerY ) );
		_zoomEnd.copy(_zoomStart);

	    } else if ( _state === STATE.PAN && !_this.noPan ) {

		_panStart.copy( getMouseOnScreen( event.layerX, event.layerY ) );
		_panEnd.copy(_panStart);

	    }

	    document.addEventListener( 'mousemove', mousemove, false );
	    document.addEventListener( 'mouseup', mouseup, false );

	    _this.dispatchEvent( startEvent );

	}

	function mousemove( event ) {

	    if ( _this.enabled === false ) return;

	    event.preventDefault();
	    event.stopPropagation();

	    if ( _state === STATE.ROTATE && !_this.noRotate ) {

		_rotateEnd.copy( getMouseProjectionOnBall( event.layerX, event.layerY ) );

	    } else if ( _state === STATE.ZOOM && !_this.noZoom ) {

		_zoomEnd.copy( getMouseOnScreen( event.layerX, event.layerY ) );

	    } else if ( _state === STATE.PAN && !_this.noPan ) {

		_panEnd.copy( getMouseOnScreen( event.layerX, event.layerY ) );

	    }

	}

	function mouseup( event ) {

	    if ( _this.enabled === false ) return;

	    event.preventDefault();
	    event.stopPropagation();

	    _state = STATE.NONE;

	    document.removeEventListener( 'mousemove', mousemove );
	    document.removeEventListener( 'mouseup', mouseup );
	    _this.dispatchEvent( endEvent );

	}

	function mousewheel( event ) {

	    if ( _this.enabled === false ) return;

	    event.preventDefault();
	    event.stopPropagation();

	    var delta = 0;

	    if ( event.wheelDelta ) { // WebKit / Opera / Explorer 9

		delta = event.wheelDelta / 40;

	    } else if ( event.detail ) { // Firefox

		delta = - event.detail / 3;

	    }

	    _zoomStart.y += delta * 0.01;
	    _this.dispatchEvent( startEvent );
	    _this.dispatchEvent( endEvent );

	}

	function touchstart( event ) {

	    if ( _this.enabled === false ) return;

	    switch ( event.touches.length ) {

	    case 1:
		_state = STATE.TOUCH_ROTATE;
		_rotateStart.copy( getMouseProjectionOnBall( event.touches[ 0 ].pageX, event.touches[ 0 ].pageY ) );
		_rotateEnd.copy( _rotateStart );
		break;

	    case 2:
		_state = STATE.TOUCH_ZOOM_PAN;
		var dx = event.touches[ 0 ].pageX - event.touches[ 1 ].pageX;
		var dy = event.touches[ 0 ].pageY - event.touches[ 1 ].pageY;
		_touchZoomDistanceEnd = _touchZoomDistanceStart = Math.sqrt( dx * dx + dy * dy );

		var x = ( event.touches[ 0 ].pageX + event.touches[ 1 ].pageX ) / 2;
		var y = ( event.touches[ 0 ].pageY + event.touches[ 1 ].pageY ) / 2;
		_panStart.copy( getMouseOnScreen( x, y ) );
		_panEnd.copy( _panStart );
		break;

	    default:
		_state = STATE.NONE;

	    }
	    _this.dispatchEvent( startEvent );


	}

	function touchmove( event ) {

	    if ( _this.enabled === false ) return;

	    event.preventDefault();
	    event.stopPropagation();

	    switch ( event.touches.length ) {

	    case 1:
		_rotateEnd.copy( getMouseProjectionOnBall( event.touches[ 0 ].pageX, event.touches[ 0 ].pageY ) );
		break;

	    case 2:
		var dx = event.touches[ 0 ].pageX - event.touches[ 1 ].pageX;
		var dy = event.touches[ 0 ].pageY - event.touches[ 1 ].pageY;
		_touchZoomDistanceEnd = Math.sqrt( dx * dx + dy * dy );

		var x = ( event.touches[ 0 ].pageX + event.touches[ 1 ].pageX ) / 2;
		var y = ( event.touches[ 0 ].pageY + event.touches[ 1 ].pageY ) / 2;
		_panEnd.copy( getMouseOnScreen( x, y ) );
		break;

	    default:
		_state = STATE.NONE;

	    }

	}

	function touchend( event ) {

	    if ( _this.enabled === false ) return;

	    switch ( event.touches.length ) {

	    case 1:
		_rotateEnd.copy( getMouseProjectionOnBall( event.touches[ 0 ].pageX, event.touches[ 0 ].pageY ) );
		_rotateStart.copy( _rotateEnd );
		break;

	    case 2:
		_touchZoomDistanceStart = _touchZoomDistanceEnd = 0;

		var x = ( event.touches[ 0 ].pageX + event.touches[ 1 ].pageX ) / 2;
		var y = ( event.touches[ 0 ].pageY + event.touches[ 1 ].pageY ) / 2;
		_panEnd.copy( getMouseOnScreen( x, y ) );
		_panStart.copy( _panEnd );
		break;

	    }

	    _state = STATE.NONE;
	    _this.dispatchEvent( endEvent );

	}

	this.domElement.addEventListener( 'contextmenu', function ( event ) { event.preventDefault(); }, false );

	this.domElement.addEventListener( 'mousedown', mousedown, false );

	this.domElement.addEventListener( 'mousewheel', mousewheel, false );
	this.domElement.addEventListener( 'DOMMouseScroll', mousewheel, false ); // firefox

	this.domElement.addEventListener( 'touchstart', touchstart, false );
	this.domElement.addEventListener( 'touchend', touchend, false );
	this.domElement.addEventListener( 'touchmove', touchmove, false );

	window.addEventListener( 'keydown', keydown, false );
	window.addEventListener( 'keyup', keyup, false );

	this.handleResize();

	// force an update at start
	this.update();

    };

    TrackballControls.prototype = Object.create(THREE.EventDispatcher.prototype);
    TrackballControls.prototype.constructor = TrackballControls;
    return TrackballControls;
});

define('utils/OrthographicTrackballControls',[], function(){
    /**
     * @author Eberhard Graether / http://egraether.com/
     * @author Mark Lundin 	/ http://mark-lundin.com
     * @author Patrick Fuller / http://patrick-fuller.com
     */

    var OrthographicTrackballControls = function ( object, domElement ) {

	var _this = this;
	var STATE = { NONE: -1, ROTATE: 0, ZOOM: 1, PAN: 2, TOUCH_ROTATE: 3, TOUCH_ZOOM_PAN: 4 };

	this.object = object;
	this.domElement = ( domElement !== undefined ) ? domElement : document;

	// API

	this.enabled = true;

	this.screen = { left: 0, top: 0, width: 0, height: 0 };

	this.rotateSpeed = 1.0;
	this.zoomSpeed = 1.2;
	this.panSpeed = 0.3;

	this.noRotate = false;
	this.noZoom = false;
	this.noPan = false;
	this.noRoll = false;

	this.staticMoving = false;
	this.dynamicDampingFactor = 0.2;

	this.keys = [ 65 /*A*/, 83 /*S*/, 68 /*D*/ ];

	// internals

	this.target = new THREE.Vector3();

	var EPS = 0.000001;

	var lastPosition = new THREE.Vector3();

	var _state = STATE.NONE,
	    _prevState = STATE.NONE,

	    _eye = new THREE.Vector3(),

	    _rotateStart = new THREE.Vector3(),
	    _rotateEnd = new THREE.Vector3(),

	    _zoomStart = new THREE.Vector2(),
	    _zoomEnd = new THREE.Vector2(),
	    _zoomFactor = 1,

	    _touchZoomDistanceStart = 0,
	    _touchZoomDistanceEnd = 0,

	    _panStart = new THREE.Vector2(),
	    _panEnd = new THREE.Vector2();

	// for reset

	this.target0 = this.target.clone();
	this.position0 = this.object.position.clone();
	this.up0 = this.object.up.clone();

	this.left0 = this.object.left;
	this.right0 = this.object.right;
	this.top0 = this.object.top;
	this.bottom0 = this.object.bottom;
	this.center0 = new THREE.Vector2((this.left0 + this.right0) / 2.0, (this.top0 + this.bottom0) / 2.0);

	// events

	var changeEvent = { type: 'change' };
	var startEvent = { type: 'start'};
	var endEvent = { type: 'end'};


	// methods

	this.handleResize = function () {

	    if ( this.domElement === document ) {

		this.screen.left = 0;
		this.screen.top = 0;
		this.screen.width = window.innerWidth;
		this.screen.height = window.innerHeight;

	    } else {

		var box = this.domElement.getBoundingClientRect();
		// adjustments come from similar code in the jquery offset() function
		var d = this.domElement.ownerDocument.documentElement;
		this.screen.left = box.left + window.pageXOffset - d.clientLeft;
		this.screen.top = box.top + window.pageYOffset - d.clientTop;
		this.screen.width = box.width;
		this.screen.height = box.height;
	    }

	    this.left0 = this.object.left;
	    this.right0 = this.object.right;
	    this.top0 = this.object.top;
	    this.bottom0 = this.object.bottom;
	    this.center0.set((this.left0 + this.right0) / 2.0, (this.top0 + this.bottom0) / 2.0);

	};

	this.handleEvent = function ( event ) {

	    if ( typeof this[ event.type ] == 'function' ) {

		this[ event.type ]( event );

	    }

	};

	var getMouseOnScreen = ( function () {

	    var vector = new THREE.Vector2();

	    return function ( pageX, pageY ) {

		vector.set(
		    ( pageX - _this.screen.left ) / _this.screen.width,
		    ( pageY - _this.screen.top ) / _this.screen.height
		);

		return vector;

	    };

	}() );

	var getMouseProjectionOnBall = ( function () {

	    var vector = new THREE.Vector3();
	    var objectUp = new THREE.Vector3();
	    var mouseOnBall = new THREE.Vector3();

	    return function ( pageX, pageY ) {

		mouseOnBall.set(
		    ( pageX - _this.screen.width * 0.5 - _this.screen.left ) / (_this.screen.width*.5),
		    ( _this.screen.height * 0.5 + _this.screen.top - pageY ) / (_this.screen.height*.5),
		    0.0
		);

		var length = mouseOnBall.length();

		if ( _this.noRoll ) {

		    if ( length < Math.SQRT1_2 ) {

			mouseOnBall.z = Math.sqrt( 1.0 - length*length );

		    } else {

			mouseOnBall.z = .5 / length;
			
		    }

		} else if ( length > 1.0 ) {

		    mouseOnBall.normalize();

		} else {

		    mouseOnBall.z = Math.sqrt( 1.0 - length * length );

		}

		_eye.copy( _this.object.position ).sub( _this.target );

		vector.copy( _this.object.up ).setLength( mouseOnBall.y );
		vector.add( objectUp.copy( _this.object.up ).cross( _eye ).setLength( mouseOnBall.x ) );
		vector.add( _eye.setLength( mouseOnBall.z ) );

		return vector;

	    };

	}() );

	this.rotateCamera = (function(){

	    var axis = new THREE.Vector3(),
		quaternion = new THREE.Quaternion();

	    return function () {

		var angle = Math.acos( _rotateStart.dot( _rotateEnd ) / _rotateStart.length() / _rotateEnd.length() );

		if ( angle ) {

		    axis.crossVectors( _rotateStart, _rotateEnd ).normalize();

		    angle *= _this.rotateSpeed;

		    quaternion.setFromAxisAngle( axis, -angle );

		    _eye.applyQuaternion( quaternion );
		    _this.object.up.applyQuaternion( quaternion );

		    _rotateEnd.applyQuaternion( quaternion );

		    if ( _this.staticMoving ) {

			_rotateStart.copy( _rotateEnd );

		    } else {

			quaternion.setFromAxisAngle( axis, angle * ( _this.dynamicDampingFactor - 1.0 ) );
			_rotateStart.applyQuaternion( quaternion );

		    }

		}
	    };

	}());

	this.zoomCamera = function () {
	    var factor;

	    if ( _state === STATE.TOUCH_ZOOM_PAN ) {

		factor = _touchZoomDistanceStart / _touchZoomDistanceEnd;
		_touchZoomDistanceStart = _touchZoomDistanceEnd;

	    } else {

		factor = 1.0 + ( _zoomEnd.y - _zoomStart.y ) * _this.zoomSpeed;

	    }

	    if ( factor !== 1.0 && factor > 0.0 ) {

		_zoomFactor *= factor;

		_this.object.left = _zoomFactor * _this.left0 + ( 1 - _zoomFactor ) *  _this.center0.x;
		_this.object.right = _zoomFactor * _this.right0 + ( 1 - _zoomFactor ) *  _this.center0.x;
		_this.object.top = _zoomFactor * _this.top0 + ( 1 - _zoomFactor ) *  _this.center0.y;
		_this.object.bottom = _zoomFactor * _this.bottom0 + ( 1 - _zoomFactor ) *  _this.center0.y;

		if ( _this.staticMoving ) {

		    _zoomStart.copy( _zoomEnd );

		} else {

		    _zoomStart.y += ( _zoomEnd.y - _zoomStart.y ) * this.dynamicDampingFactor;

		}

	    }

	};

	this.panCamera = (function(){

	    var mouseChange = new THREE.Vector2(),
		objectUp = new THREE.Vector3(),
		pan = new THREE.Vector3();

	    return function () {

		mouseChange.copy( _panEnd ).sub( _panStart );

		if ( mouseChange.lengthSq() ) {

		    mouseChange.multiplyScalar( _eye.length() * _this.panSpeed );

		    pan.copy( _eye ).cross( _this.object.up ).setLength( mouseChange.x );
		    pan.add( objectUp.copy( _this.object.up ).setLength( mouseChange.y ) );

		    _this.object.position.add( pan );
		    _this.target.add( pan );

		    if ( _this.staticMoving ) {

			_panStart.copy( _panEnd );

		    } else {

			_panStart.add( mouseChange.subVectors( _panEnd, _panStart ).multiplyScalar( _this.dynamicDampingFactor ) );

		    }

		}
	    };

	}());

	this.update = function () {

	    _eye.subVectors( _this.object.position, _this.target );

	    if ( !_this.noRotate ) {

		_this.rotateCamera();

	    }

	    if ( !_this.noZoom ) {

		_this.zoomCamera();
		_this.object.updateProjectionMatrix();

	    }

	    if ( !_this.noPan ) {

		_this.panCamera();

	    }

	    _this.object.position.addVectors( _this.target, _eye );

	    _this.object.lookAt( _this.target );

	    if ( lastPosition.distanceToSquared( _this.object.position ) > EPS ) {

		_this.dispatchEvent( changeEvent );

		lastPosition.copy( _this.object.position );

	    }

	};

	this.reset = function () {

	    _state = STATE.NONE;
	    _prevState = STATE.NONE;

	    _this.target.copy( _this.target0 );
	    _this.object.position.copy( _this.position0 );
	    _this.object.up.copy( _this.up0 );

	    _eye.subVectors( _this.object.position, _this.target );

	    _this.object.left = _this.left0;
	    _this.object.right = _this.right0;
	    _this.object.top = _this.top0;
	    _this.object.bottom = _this.bottom0;

	    _this.object.lookAt( _this.target );

	    _this.dispatchEvent( changeEvent );

	    lastPosition.copy( _this.object.position );

	};

	// listeners

	function keydown( event ) {

	    if ( _this.enabled === false ) return;

	    window.removeEventListener( 'keydown', keydown );

	    _prevState = _state;

	    if ( _state !== STATE.NONE ) {

		return;

	    } else if ( event.keyCode === _this.keys[ STATE.ROTATE ] && !_this.noRotate ) {

		_state = STATE.ROTATE;

	    } else if ( event.keyCode === _this.keys[ STATE.ZOOM ] && !_this.noZoom ) {

		_state = STATE.ZOOM;

	    } else if ( event.keyCode === _this.keys[ STATE.PAN ] && !_this.noPan ) {

		_state = STATE.PAN;

	    }

	}

	function keyup( event ) {

	    if ( _this.enabled === false ) return;

	    _state = _prevState;

	    window.addEventListener( 'keydown', keydown, false );

	}

	function mousedown( event ) {

	    if ( _this.enabled === false ) return;

	    event.preventDefault();
	    event.stopPropagation();

	    if ( _state === STATE.NONE ) {

		_state = event.button;

	    }

	    if ( _state === STATE.ROTATE && !_this.noRotate ) {

		_rotateStart.copy( getMouseProjectionOnBall( event.pageX, event.pageY ) );
		_rotateEnd.copy( _rotateStart );

	    } else if ( _state === STATE.ZOOM && !_this.noZoom ) {

		_zoomStart.copy( getMouseOnScreen( event.pageX, event.pageY ) );
		_zoomEnd.copy(_zoomStart);

	    } else if ( _state === STATE.PAN && !_this.noPan ) {

		_panStart.copy( getMouseOnScreen( event.pageX, event.pageY ) );
		_panEnd.copy(_panStart);

	    }

	    document.addEventListener( 'mousemove', mousemove, false );
	    document.addEventListener( 'mouseup', mouseup, false );

	    _this.dispatchEvent( startEvent );

	}

	function mousemove( event ) {

	    if ( _this.enabled === false ) return;

	    event.preventDefault();
	    event.stopPropagation();

	    if ( _state === STATE.ROTATE && !_this.noRotate ) {

		_rotateEnd.copy( getMouseProjectionOnBall( event.pageX, event.pageY ) );

	    } else if ( _state === STATE.ZOOM && !_this.noZoom ) {

		_zoomEnd.copy( getMouseOnScreen( event.pageX, event.pageY ) );

	    } else if ( _state === STATE.PAN && !_this.noPan ) {

		_panEnd.copy( getMouseOnScreen( event.pageX, event.pageY ) );

	    }

	}

	function mouseup( event ) {

	    if ( _this.enabled === false ) return;

	    event.preventDefault();
	    event.stopPropagation();

	    _state = STATE.NONE;

	    document.removeEventListener( 'mousemove', mousemove );
	    document.removeEventListener( 'mouseup', mouseup );
	    _this.dispatchEvent( endEvent );

	}

	function mousewheel( event ) {

	    if ( _this.enabled === false ) return;

	    event.preventDefault();
	    event.stopPropagation();

	    var delta = 0;

	    if ( event.wheelDelta ) { // WebKit / Opera / Explorer 9

		delta = event.wheelDelta / 40;

	    } else if ( event.detail ) { // Firefox

		delta = - event.detail / 3;

	    }

	    _zoomStart.y += delta * 0.01;
	    _this.dispatchEvent( startEvent );
	    _this.dispatchEvent( endEvent );

	}

	function touchstart( event ) {

	    if ( _this.enabled === false ) return;

	    switch ( event.touches.length ) {

	    case 1:
		_state = STATE.TOUCH_ROTATE;
		_rotateStart.copy( getMouseProjectionOnBall( event.touches[ 0 ].pageX, event.touches[ 0 ].pageY ) );
		_rotateEnd.copy( _rotateStart );
		break;

	    case 2:
		_state = STATE.TOUCH_ZOOM_PAN;
		var dx = event.touches[ 0 ].pageX - event.touches[ 1 ].pageX;
		var dy = event.touches[ 0 ].pageY - event.touches[ 1 ].pageY;
		_touchZoomDistanceEnd = _touchZoomDistanceStart = Math.sqrt( dx * dx + dy * dy );

		var x = ( event.touches[ 0 ].pageX + event.touches[ 1 ].pageX ) / 2;
		var y = ( event.touches[ 0 ].pageY + event.touches[ 1 ].pageY ) / 2;
		_panStart.copy( getMouseOnScreen( x, y ) );
		_panEnd.copy( _panStart );
		break;

	    default:
		_state = STATE.NONE;

	    }
	    _this.dispatchEvent( startEvent );


	}

	function touchmove( event ) {

	    if ( _this.enabled === false ) return;

	    event.preventDefault();
	    event.stopPropagation();

	    switch ( event.touches.length ) {

	    case 1:
		_rotateEnd.copy( getMouseProjectionOnBall( event.touches[ 0 ].pageX, event.touches[ 0 ].pageY ) );
		break;

	    case 2:
		var dx = event.touches[ 0 ].pageX - event.touches[ 1 ].pageX;
		var dy = event.touches[ 0 ].pageY - event.touches[ 1 ].pageY;
		_touchZoomDistanceEnd = Math.sqrt( dx * dx + dy * dy );

		var x = ( event.touches[ 0 ].pageX + event.touches[ 1 ].pageX ) / 2;
		var y = ( event.touches[ 0 ].pageY + event.touches[ 1 ].pageY ) / 2;
		_panEnd.copy( getMouseOnScreen( x, y ) );
		break;

	    default:
		_state = STATE.NONE;

	    }

	}

	function touchend( event ) {

	    if ( _this.enabled === false ) return;

	    switch ( event.touches.length ) {

	    case 1:
		_rotateEnd.copy( getMouseProjectionOnBall( event.touches[ 0 ].pageX, event.touches[ 0 ].pageY ) );
		_rotateStart.copy( _rotateEnd );
		break;

	    case 2:
		_touchZoomDistanceStart = _touchZoomDistanceEnd = 0;

		var x = ( event.touches[ 0 ].pageX + event.touches[ 1 ].pageX ) / 2;
		var y = ( event.touches[ 0 ].pageY + event.touches[ 1 ].pageY ) / 2;
		_panEnd.copy( getMouseOnScreen( x, y ) );
		_panStart.copy( _panEnd );
		break;

	    }

	    _state = STATE.NONE;
	    _this.dispatchEvent( endEvent );

	}

	this.domElement.addEventListener( 'contextmenu', function ( event ) { event.preventDefault(); }, false );

	this.domElement.addEventListener( 'mousedown', mousedown, false );

	this.domElement.addEventListener( 'mousewheel', mousewheel, false );
	this.domElement.addEventListener( 'DOMMouseScroll', mousewheel, false ); // firefox

	this.domElement.addEventListener( 'touchstart', touchstart, false );
	this.domElement.addEventListener( 'touchend', touchend, false );
	this.domElement.addEventListener( 'touchmove', touchmove, false );

	window.addEventListener( 'keydown', keydown, false );
	window.addEventListener( 'keyup', keyup, false );

	this.handleResize();

	// force an update at start
	this.update();

    };

    OrthographicTrackballControls.prototype = Object.create( THREE.EventDispatcher.prototype );
    OrthographicTrackballControls.prototype.constructor = OrthographicTrackballControls;
    return OrthographicTrackballControls;
});

// http://threejs.org/examples/js/controls/OrbitControls.js
define('utils/OrbitControls',[],function(){

	function OrbitConstraint ( object ) {

		this.object = object;

		// "target" sets the location of focus, where the object orbits around
		// and where it pans with respect to.
		this.target = new THREE.Vector3();

		// Limits to how far you can dolly in and out ( PerspectiveCamera only )
		this.minDistance = 0;
		this.maxDistance = Infinity;

		// Limits to how far you can zoom in and out ( OrthographicCamera only )
		this.minZoom = 0;
		this.maxZoom = Infinity;

		// How far you can orbit vertically, upper and lower limits.
		// Range is 0 to Math.PI radians.
		this.minPolarAngle = 0; // radians
		this.maxPolarAngle = Math.PI; // radians

		// How far you can orbit horizontally, upper and lower limits.
		// If set, must be a sub-interval of the interval [ - Math.PI, Math.PI ].
		this.minAzimuthAngle = - Infinity; // radians
		this.maxAzimuthAngle = Infinity; // radians

		// Set to true to enable damping (inertia)
		// If damping is enabled, you must call controls.update() in your animation loop
		this.enableDamping = false;
		this.dampingFactor = 0.25;

		////////////
		// internals

		var scope = this;

		var EPS = 0.000001;

		// Current position in spherical coordinate system.
		var theta;
		var phi;

		// Pending changes
		var phiDelta = 0;
		var thetaDelta = 0;
		var scale = 1;
		var panOffset = new THREE.Vector3();
		var zoomChanged = false;

		// API

		this.getPolarAngle = function () {

			return phi;

		};

		this.getAzimuthalAngle = function () {

			return theta;

		};

		this.rotateLeft = function ( angle ) {

			thetaDelta -= angle;

		};

		this.rotateUp = function ( angle ) {

			phiDelta -= angle;

		};

		// pass in distance in world space to move left
		this.panLeft = function() {

			var v = new THREE.Vector3();

			return function panLeft ( distance ) {

				var te = this.object.matrix.elements;

				// get X column of matrix
				v.set( te[ 0 ], te[ 1 ], te[ 2 ] );
				v.multiplyScalar( - distance );

				panOffset.add( v );

			};

		}();

		// pass in distance in world space to move up
		this.panUp = function() {

			var v = new THREE.Vector3();

			return function panUp ( distance ) {

				var te = this.object.matrix.elements;

				// get Y column of matrix
				v.set( te[ 4 ], te[ 5 ], te[ 6 ] );
				v.multiplyScalar( distance );

				panOffset.add( v );

			};

		}();

		// pass in x,y of change desired in pixel space,
		// right and down are positive
		this.pan = function ( deltaX, deltaY, screenWidth, screenHeight ) {

			if ( scope.object instanceof THREE.PerspectiveCamera ) {

				// perspective
				var position = scope.object.position;
				var offset = position.clone().sub( scope.target );
				var targetDistance = offset.length();

				// half of the fov is center to top of screen
				targetDistance *= Math.tan( ( scope.object.fov / 2 ) * Math.PI / 180.0 );

				// we actually don't use screenWidth, since perspective camera is fixed to screen height
				scope.panLeft( 2 * deltaX * targetDistance / screenHeight );
				scope.panUp( 2 * deltaY * targetDistance / screenHeight );

			} else if ( scope.object instanceof THREE.OrthographicCamera ) {

				// orthographic
				scope.panLeft( deltaX * ( scope.object.right - scope.object.left ) / screenWidth );
				scope.panUp( deltaY * ( scope.object.top - scope.object.bottom ) / screenHeight );

			} else {

				// camera neither orthographic or perspective
				console.warn( 'WARNING: OrbitControls.js encountered an unknown camera type - pan disabled.' );

			}

		};

		this.dollyIn = function ( dollyScale ) {

			if ( scope.object instanceof THREE.PerspectiveCamera ) {

				scale /= dollyScale;

			} else if ( scope.object instanceof THREE.OrthographicCamera ) {

				scope.object.zoom = Math.max( this.minZoom, Math.min( this.maxZoom, this.object.zoom * dollyScale ) );
				scope.object.updateProjectionMatrix();
				zoomChanged = true;

			} else {

				console.warn( 'WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled.' );

			}

		};

		this.dollyOut = function ( dollyScale ) {

			if ( scope.object instanceof THREE.PerspectiveCamera ) {

				scale *= dollyScale;

			} else if ( scope.object instanceof THREE.OrthographicCamera ) {

				scope.object.zoom = Math.max( this.minZoom, Math.min( this.maxZoom, this.object.zoom / dollyScale ) );
				scope.object.updateProjectionMatrix();
				zoomChanged = true;

			} else {

				console.warn( 'WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled.' );

			}

		};

		this.update = function() {

			var offset = new THREE.Vector3();

			// so camera.up is the orbit axis
			var quat = new THREE.Quaternion().setFromUnitVectors( object.up, new THREE.Vector3( 0, 1, 0 ) );
			var quatInverse = quat.clone().inverse();

			var lastPosition = new THREE.Vector3();
			var lastQuaternion = new THREE.Quaternion();

			return function () {

				var position = this.object.position;

				offset.copy( position ).sub( this.target );

				// rotate offset to "y-axis-is-up" space
				offset.applyQuaternion( quat );

				// angle from z-axis around y-axis

				theta = Math.atan2( offset.x, offset.z );

				// angle from y-axis

				phi = Math.atan2( Math.sqrt( offset.x * offset.x + offset.z * offset.z ), offset.y );

				theta += thetaDelta;
				phi += phiDelta;

				// restrict theta to be between desired limits
				theta = Math.max( this.minAzimuthAngle, Math.min( this.maxAzimuthAngle, theta ) );

				// restrict phi to be between desired limits
				phi = Math.max( this.minPolarAngle, Math.min( this.maxPolarAngle, phi ) );

				// restrict phi to be betwee EPS and PI-EPS
				phi = Math.max( EPS, Math.min( Math.PI - EPS, phi ) );

				var radius = offset.length() * scale;

				// restrict radius to be between desired limits
				radius = Math.max( this.minDistance, Math.min( this.maxDistance, radius ) );

				// move target to panned location
				this.target.add( panOffset );

				offset.x = radius * Math.sin( phi ) * Math.sin( theta );
				offset.y = radius * Math.cos( phi );
				offset.z = radius * Math.sin( phi ) * Math.cos( theta );

				// rotate offset back to "camera-up-vector-is-up" space
				offset.applyQuaternion( quatInverse );

				position.copy( this.target ).add( offset );

				this.object.lookAt( this.target );

				if ( this.enableDamping === true ) {

					thetaDelta *= ( 1 - this.dampingFactor );
					phiDelta *= ( 1 - this.dampingFactor );

				} else {

					thetaDelta = 0;
					phiDelta = 0;

				}

				scale = 1;
				panOffset.set( 0, 0, 0 );

				// update condition is:
				// min(camera displacement, camera rotation in radians)^2 > EPS
				// using small-angle approximation cos(x/2) = 1 - x^2 / 8

				if ( zoomChanged ||
					 lastPosition.distanceToSquared( this.object.position ) > EPS ||
				    8 * ( 1 - lastQuaternion.dot( this.object.quaternion ) ) > EPS ) {

					lastPosition.copy( this.object.position );
					lastQuaternion.copy( this.object.quaternion );
					zoomChanged = false;

					return true;

				}

				return false;

			};

		}();

	};


	// This set of controls performs orbiting, dollying (zooming), and panning. It maintains
	// the "up" direction as +Y, unlike the TrackballControls. Touch on tablet and phones is
	// supported.
	//
	//    Orbit - left mouse / touch: one finger move
	//    Zoom - middle mouse, or mousewheel / touch: two finger spread or squish
	//    Pan - right mouse, or arrow keys / touch: three finter swipe

	var OrbitControls = function ( object, domElement ) {

		var constraint = new OrbitConstraint( object );

		this.domElement = ( domElement !== undefined ) ? domElement : document;

		// API

		Object.defineProperty( this, 'constraint', {

			get: function() {

				return constraint;

			}

		} );

		this.getPolarAngle = function () {

			return constraint.getPolarAngle();

		};

		this.getAzimuthalAngle = function () {

			return constraint.getAzimuthalAngle();

		};

		// Set to false to disable this control
		this.enabled = true;

		// center is old, deprecated; use "target" instead
		this.center = this.target;

		// This option actually enables dollying in and out; left as "zoom" for
		// backwards compatibility.
		// Set to false to disable zooming
		this.enableZoom = true;
		this.zoomSpeed = 1.0;

		// Set to false to disable rotating
		this.enableRotate = true;
		this.rotateSpeed = 1.0;

		// Set to false to disable panning
		this.enablePan = true;
		this.keyPanSpeed = 7.0;	// pixels moved per arrow key push

		// Set to true to automatically rotate around the target
		// If auto-rotate is enabled, you must call controls.update() in your animation loop
		this.autoRotate = false;
		this.autoRotateSpeed = 2.0; // 30 seconds per round when fps is 60

		// Set to false to disable use of the keys
		this.enableKeys = true;

		// The four arrow keys
		this.keys = { LEFT: 37, UP: 38, RIGHT: 39, BOTTOM: 40 };

		// Mouse buttons
		this.mouseButtons = { ORBIT: THREE.MOUSE.LEFT, ZOOM: THREE.MOUSE.MIDDLE, PAN: THREE.MOUSE.RIGHT };

		////////////
		// internals

		var scope = this;

		var rotateStart = new THREE.Vector2();
		var rotateEnd = new THREE.Vector2();
		var rotateDelta = new THREE.Vector2();

		var panStart = new THREE.Vector2();
		var panEnd = new THREE.Vector2();
		var panDelta = new THREE.Vector2();

		var dollyStart = new THREE.Vector2();
		var dollyEnd = new THREE.Vector2();
		var dollyDelta = new THREE.Vector2();

		var STATE = { NONE : - 1, ROTATE : 0, DOLLY : 1, PAN : 2, TOUCH_ROTATE : 3, TOUCH_DOLLY : 4, TOUCH_PAN : 5 };

		var state = STATE.NONE;

		// for reset
        console.log(this.target);
		this.target0 = this.target.clone();
		this.position0 = this.object.position.clone();
		this.zoom0 = this.object.zoom;

		// events

		var changeEvent = { type: 'change' };
		var startEvent = { type: 'start' };
		var endEvent = { type: 'end' };

		// pass in x,y of change desired in pixel space,
		// right and down are positive
		function pan( deltaX, deltaY ) {

			var element = scope.domElement === document ? scope.domElement.body : scope.domElement;

			constraint.pan( deltaX, deltaY, element.clientWidth, element.clientHeight );

		}

		this.update = function () {

			if ( this.autoRotate && state === STATE.NONE ) {

				constraint.rotateLeft( getAutoRotationAngle() );

			}

			if ( constraint.update() === true ) {

				this.dispatchEvent( changeEvent );

			}

		};

		this.reset = function () {

			state = STATE.NONE;

			this.target.copy( this.target0 );
			this.object.position.copy( this.position0 );
			this.object.zoom = this.zoom0;

			this.object.updateProjectionMatrix();
			this.dispatchEvent( changeEvent );

			this.update();

		};

		function getAutoRotationAngle() {

			return 2 * Math.PI / 60 / 60 * scope.autoRotateSpeed;

		}

		function getZoomScale() {

			return Math.pow( 0.95, scope.zoomSpeed );

		}

		function onMouseDown( event ) {

			if ( scope.enabled === false ) return;

			event.preventDefault();

			if ( event.button === scope.mouseButtons.ORBIT ) {

				if ( scope.enableRotate === false ) return;

				state = STATE.ROTATE;

				rotateStart.set( event.clientX, event.clientY );

			} else if ( event.button === scope.mouseButtons.ZOOM ) {

				if ( scope.enableZoom === false ) return;

				state = STATE.DOLLY;

				dollyStart.set( event.clientX, event.clientY );

			} else if ( event.button === scope.mouseButtons.PAN ) {

				if ( scope.enablePan === false ) return;

				state = STATE.PAN;

				panStart.set( event.clientX, event.clientY );

			}

			if ( state !== STATE.NONE ) {

				document.addEventListener( 'mousemove', onMouseMove, false );
				document.addEventListener( 'mouseup', onMouseUp, false );
				scope.dispatchEvent( startEvent );

			}

		}

		function onMouseMove( event ) {

			if ( scope.enabled === false ) return;

			event.preventDefault();

			var element = scope.domElement === document ? scope.domElement.body : scope.domElement;

			if ( state === STATE.ROTATE ) {

				if ( scope.enableRotate === false ) return;

				rotateEnd.set( event.clientX, event.clientY );
				rotateDelta.subVectors( rotateEnd, rotateStart );

				// rotating across whole screen goes 360 degrees around
				constraint.rotateLeft( 2 * Math.PI * rotateDelta.x / element.clientWidth * scope.rotateSpeed );

				// rotating up and down along whole screen attempts to go 360, but limited to 180
				constraint.rotateUp( 2 * Math.PI * rotateDelta.y / element.clientHeight * scope.rotateSpeed );

				rotateStart.copy( rotateEnd );

			} else if ( state === STATE.DOLLY ) {

				if ( scope.enableZoom === false ) return;

				dollyEnd.set( event.clientX, event.clientY );
				dollyDelta.subVectors( dollyEnd, dollyStart );

				if ( dollyDelta.y > 0 ) {

					constraint.dollyIn( getZoomScale() );

				} else if ( dollyDelta.y < 0 ) {

					constraint.dollyOut( getZoomScale() );

				}

				dollyStart.copy( dollyEnd );

			} else if ( state === STATE.PAN ) {

				if ( scope.enablePan === false ) return;

				panEnd.set( event.clientX, event.clientY );
				panDelta.subVectors( panEnd, panStart );

				pan( panDelta.x, panDelta.y );

				panStart.copy( panEnd );

			}

			if ( state !== STATE.NONE ) scope.update();

		}

		function onMouseUp( /* event */ ) {

			if ( scope.enabled === false ) return;

			document.removeEventListener( 'mousemove', onMouseMove, false );
			document.removeEventListener( 'mouseup', onMouseUp, false );
			scope.dispatchEvent( endEvent );
			state = STATE.NONE;

		}

		function onMouseWheel( event ) {

			if ( scope.enabled === false || scope.enableZoom === false || state !== STATE.NONE ) return;

			event.preventDefault();
			event.stopPropagation();

			var delta = 0;

			if ( event.wheelDelta !== undefined ) {

				// WebKit / Opera / Explorer 9

				delta = event.wheelDelta;

			} else if ( event.detail !== undefined ) {

				// Firefox

				delta = - event.detail;

			}

			if ( delta > 0 ) {

				constraint.dollyOut( getZoomScale() );

			} else if ( delta < 0 ) {

				constraint.dollyIn( getZoomScale() );

			}

			scope.update();
			scope.dispatchEvent( startEvent );
			scope.dispatchEvent( endEvent );

		}

		function onKeyDown( event ) {

			if ( scope.enabled === false || scope.enableKeys === false || scope.enablePan === false ) return;

			switch ( event.keyCode ) {

				case scope.keys.UP:
					pan( 0, scope.keyPanSpeed );
					scope.update();
					break;

				case scope.keys.BOTTOM:
					pan( 0, - scope.keyPanSpeed );
					scope.update();
					break;

				case scope.keys.LEFT:
					pan( scope.keyPanSpeed, 0 );
					scope.update();
					break;

				case scope.keys.RIGHT:
					pan( - scope.keyPanSpeed, 0 );
					scope.update();
					break;

			}

		}

		function touchstart( event ) {

			if ( scope.enabled === false ) return;

			switch ( event.touches.length ) {

				case 1:	// one-fingered touch: rotate

					if ( scope.enableRotate === false ) return;

					state = STATE.TOUCH_ROTATE;

					rotateStart.set( event.touches[ 0 ].pageX, event.touches[ 0 ].pageY );
					break;

				case 2:	// two-fingered touch: dolly

					if ( scope.enableZoom === false ) return;

					state = STATE.TOUCH_DOLLY;

					var dx = event.touches[ 0 ].pageX - event.touches[ 1 ].pageX;
					var dy = event.touches[ 0 ].pageY - event.touches[ 1 ].pageY;
					var distance = Math.sqrt( dx * dx + dy * dy );
					dollyStart.set( 0, distance );
					break;

				case 3: // three-fingered touch: pan

					if ( scope.enablePan === false ) return;

					state = STATE.TOUCH_PAN;

					panStart.set( event.touches[ 0 ].pageX, event.touches[ 0 ].pageY );
					break;

				default:

					state = STATE.NONE;

			}

			if ( state !== STATE.NONE ) scope.dispatchEvent( startEvent );

		}

		function touchmove( event ) {

			if ( scope.enabled === false ) return;

			event.preventDefault();
			event.stopPropagation();

			var element = scope.domElement === document ? scope.domElement.body : scope.domElement;

			switch ( event.touches.length ) {

				case 1: // one-fingered touch: rotate

					if ( scope.enableRotate === false ) return;
					if ( state !== STATE.TOUCH_ROTATE ) return;

					rotateEnd.set( event.touches[ 0 ].pageX, event.touches[ 0 ].pageY );
					rotateDelta.subVectors( rotateEnd, rotateStart );

					// rotating across whole screen goes 360 degrees around
					constraint.rotateLeft( 2 * Math.PI * rotateDelta.x / element.clientWidth * scope.rotateSpeed );
					// rotating up and down along whole screen attempts to go 360, but limited to 180
					constraint.rotateUp( 2 * Math.PI * rotateDelta.y / element.clientHeight * scope.rotateSpeed );

					rotateStart.copy( rotateEnd );

					scope.update();
					break;

				case 2: // two-fingered touch: dolly

					if ( scope.enableZoom === false ) return;
					if ( state !== STATE.TOUCH_DOLLY ) return;

					var dx = event.touches[ 0 ].pageX - event.touches[ 1 ].pageX;
					var dy = event.touches[ 0 ].pageY - event.touches[ 1 ].pageY;
					var distance = Math.sqrt( dx * dx + dy * dy );

					dollyEnd.set( 0, distance );
					dollyDelta.subVectors( dollyEnd, dollyStart );

					if ( dollyDelta.y > 0 ) {

						constraint.dollyOut( getZoomScale() );

					} else if ( dollyDelta.y < 0 ) {

						constraint.dollyIn( getZoomScale() );

					}

					dollyStart.copy( dollyEnd );

					scope.update();
					break;

				case 3: // three-fingered touch: pan

					if ( scope.enablePan === false ) return;
					if ( state !== STATE.TOUCH_PAN ) return;

					panEnd.set( event.touches[ 0 ].pageX, event.touches[ 0 ].pageY );
					panDelta.subVectors( panEnd, panStart );

					pan( panDelta.x, panDelta.y );

					panStart.copy( panEnd );

					scope.update();
					break;

				default:

					state = STATE.NONE;

			}

		}

		function touchend( /* event */ ) {

			if ( scope.enabled === false ) return;

			scope.dispatchEvent( endEvent );
			state = STATE.NONE;

		}

		function contextmenu( event ) {

			event.preventDefault();

		}

		this.dispose = function() {

			this.domElement.removeEventListener( 'contextmenu', contextmenu, false );
			this.domElement.removeEventListener( 'mousedown', onMouseDown, false );
			this.domElement.removeEventListener( 'mousewheel', onMouseWheel, false );
			this.domElement.removeEventListener( 'DOMMouseScroll', onMouseWheel, false ); // firefox

			this.domElement.removeEventListener( 'touchstart', touchstart, false );
			this.domElement.removeEventListener( 'touchend', touchend, false );
			this.domElement.removeEventListener( 'touchmove', touchmove, false );

			document.removeEventListener( 'mousemove', onMouseMove, false );
			document.removeEventListener( 'mouseup', onMouseUp, false );

			window.removeEventListener( 'keydown', onKeyDown, false );

		}

		this.domElement.addEventListener( 'contextmenu', contextmenu, false );

		this.domElement.addEventListener( 'mousedown', onMouseDown, false );
		this.domElement.addEventListener( 'mousewheel', onMouseWheel, false );
		this.domElement.addEventListener( 'DOMMouseScroll', onMouseWheel, false ); // firefox

		this.domElement.addEventListener( 'touchstart', touchstart, false );
		this.domElement.addEventListener( 'touchend', touchend, false );
		this.domElement.addEventListener( 'touchmove', touchmove, false );

		window.addEventListener( 'keydown', onKeyDown, false );

		// force an update at start
		this.update();

	};

	OrbitControls.prototype = Object.create( THREE.EventDispatcher.prototype );
	OrbitControls.prototype.constructor = OrbitControls;

	Object.defineProperties( OrbitControls.prototype, {

		object: {

			get: function () {

				return this.constraint.object;

			}

		},

		target: {

			get: function () {

				return this.constraint.target;

			},

			set: function ( value ) {

				console.warn( 'OrbitControls: target is now immutable. Use target.set() instead.' );
				this.constraint.target.copy( value );

			}

		},

		minDistance : {

			get: function () {

				return this.constraint.minDistance;

			},

			set: function ( value ) {

				this.constraint.minDistance = value;

			}

		},

		maxDistance : {

			get: function () {

				return this.constraint.maxDistance;

			},

			set: function ( value ) {

				this.constraint.maxDistance = value;

			}

		},

		minZoom : {

			get: function () {

				return this.constraint.minZoom;

			},

			set: function ( value ) {

				this.constraint.minZoom = value;

			}

		},

		maxZoom : {

			get: function () {

				return this.constraint.maxZoom;

			},

			set: function ( value ) {

				this.constraint.maxZoom = value;

			}

		},

		minPolarAngle : {

			get: function () {

				return this.constraint.minPolarAngle;

			},

			set: function ( value ) {

				this.constraint.minPolarAngle = value;

			}

		},

		maxPolarAngle : {

			get: function () {

				return this.constraint.maxPolarAngle;

			},

			set: function ( value ) {

				this.constraint.maxPolarAngle = value;

			}

		},

		minAzimuthAngle : {

			get: function () {

				return this.constraint.minAzimuthAngle;

			},

			set: function ( value ) {

				this.constraint.minAzimuthAngle = value;

			}

		},

		maxAzimuthAngle : {

			get: function () {

				return this.constraint.maxAzimuthAngle;

			},

			set: function ( value ) {

				this.constraint.maxAzimuthAngle = value;

			}

		},

		enableDamping : {

			get: function () {

				return this.constraint.enableDamping;

			},

			set: function ( value ) {

				this.constraint.enableDamping = value;

			}

		},

		dampingFactor : {

			get: function () {

				return this.constraint.dampingFactor;

			},

			set: function ( value ) {

				this.constraint.dampingFactor = value;

			}

		},

		// backward compatibility

		noZoom: {

			get: function () {

				console.warn( 'OrbitControls: .noZoom has been deprecated. Use .enableZoom instead.' );
				return ! this.enableZoom;

			},

			set: function ( value ) {

				console.warn( 'OrbitControls: .noZoom has been deprecated. Use .enableZoom instead.' );
				this.enableZoom = ! value;

			}

		},

		noRotate: {

			get: function () {

				console.warn( 'OrbitControls: .noRotate has been deprecated. Use .enableRotate instead.' );
				return ! this.enableRotate;

			},

			set: function ( value ) {

				console.warn( 'OrbitControls: .noRotate has been deprecated. Use .enableRotate instead.' );
				this.enableRotate = ! value;

			}

		},

		noPan: {

			get: function () {

				console.warn( 'OrbitControls: .noPan has been deprecated. Use .enablePan instead.' );
				return ! this.enablePan;

			},

			set: function ( value ) {

				console.warn( 'OrbitControls: .noPan has been deprecated. Use .enablePan instead.' );
				this.enablePan = ! value;

			}

		},

		noKeys: {

			get: function () {

				console.warn( 'OrbitControls: .noKeys has been deprecated. Use .enableKeys instead.' );
				return ! this.enableKeys;

			},

			set: function ( value ) {

				console.warn( 'OrbitControls: .noKeys has been deprecated. Use .enableKeys instead.' );
				this.enableKeys = ! value;

			}

		},

		staticMoving : {

			get: function () {

				console.warn( 'OrbitControls: .staticMoving has been deprecated. Use .enableDamping instead.' );
				return ! this.constraint.enableDamping;

			},

			set: function ( value ) {

				console.warn( 'OrbitControls: .staticMoving has been deprecated. Use .enableDamping instead.' );
				this.constraint.enableDamping = ! value;

			}

		},

		dynamicDampingFactor : {

			get: function () {

				console.warn( 'OrbitControls: .dynamicDampingFactor has been renamed. Use .dampingFactor instead.' );
				return this.constraint.dampingFactor;

			},

			set: function ( value ) {

				console.warn( 'OrbitControls: .dynamicDampingFactor has been renamed. Use .dampingFactor instead.' );
				this.constraint.dampingFactor = value;

			}

		}

	} );

    return OrbitControls;
});

define('utils/utils',[],function(){
    var mixin = function(sub, sup) {
	sup.call(sub);
    };

    var merge = function(dest, src){
	for(var key in src){
	    dest[key] = src[key];
	}
	return dest;
    };

    var exports = {
	mixin:mixin,
	merge:merge
    };

    return exports;
});

define('components/world',[
    "utils/TrackballControls",
    "utils/OrthographicTrackballControls",
    "utils/OrbitControls",
    "utils/utils"
],function(TrackballControls, OrthographicTrackballControls, OrbitControls, Utils){

    function World(options){
	this.options = {
	    width: 0,
	    height: 0,
	    perspective: true,
	    bg_color: 0xffffff,
        orbit: false,
	    save_image: false
	};

	if(arguments.length > 0){
	    Utils.merge(this.options, options);
	};

	this.scene = new THREE.Scene();

	if(this.options.perspective)
	    this.camera = new THREE.PerspectiveCamera(45, this.options.width/this.options.height, 1, 1000);
	else
	    this.camera = new THREE.OrthographicCamera(-20,20,-20,20);

	this.scene.add(this.camera);

	var positions = [[1,1,1],[-1,-1,1],[-1,1,1],[1,-1,1]];
	for(var i=0;i<4;i++){
	    var light=new THREE.DirectionalLight(0xdddddd);
	    light.position.set(positions[i][0],positions[i][1],1*positions[i][2]);
	    this.scene.add(light);
	}

	this.renderer = new THREE.WebGLRenderer({
	    antialias:true, 
	    clearAlpha: 1,
	    preserveDrawingBuffer: !this.options.save_image
	});

	this.renderer.setSize(this.options.width, this.options.height);
	this.renderer.setClearColor(this.options.bg_color, 1);
	
	this.renderer.sortObjects = false;

	if(this.options.perspective && this.options.orbit)
	    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    else if (this.options.perspective)
        this.controls = new TrackballControls(this.camera, this.renderer.domElement);
	else
	    this.controls = new OrthographicTrackballControls(this.camera, this.renderer.domElement);

	this.controls.screen = {left: 0, top: 0, width: this.options.width, height: this.options.height};
	this.controls.rotateSpeed = 0.5;

	this.camera.position.set(-30, 31,42);
	this.camera.rotation.set(-0.6,-0.5,0.6);

	return this;
    }

    World.prototype.begin = function(selection){
	selection[0][0].appendChild(this.renderer.domElement);
	var world = this;
        var interval = 1000/30;
        var before = Date.now();

	this.animate = function(){
	    window.requestAnimationFrame(world.animate);
            var now = Date.now();
            if(now - before > interval){
                before = now;
	        world.renderer.render(world.scene, world.camera);
	        world.controls.update();
            }
	};

	this.animate();
    };

    World.prototype.addMesh = function(mesh){
	if(mesh instanceof Array){
	    for(var i=0; i<mesh.length; i++){
		this.scene.add(mesh[i]);
	    }
	}else{
	    this.scene.add(mesh);
	}
    };

    World.prototype.removeMesh = function(mesh){
        if(mesh instanceof Array){
	    for(var i=0; i<mesh.length; i++){
		this.scene.remove(mesh[i]);
	    }
        }else{
	    this.scene.remove(mesh);
        }
    };

    return World;
});

define('components/space',[
    "utils/utils"
],function(Utils){
    function Space(ranges, options){
	this.options = {
	    axis_labels: {x:"Hey", y:"Yo", z:"Z"},
	    mode: 'wireframe',
	    grid: true
	};

	if(arguments.length > 1){
	    Utils.merge(this.options, options);
	};
	
	var BIGIN=-10, END=10, WIDTH=END-BIGIN;
	var geometry = new THREE.PlaneGeometry(WIDTH,WIDTH);
	var material = new THREE.MeshBasicMaterial({color:0xf0f0f0, shading: THREE.FlatShading, overdraw: 0.5, side: THREE.DoubleSide});
	var newV = function(x,y,z){return new THREE.Vector3(x,y,z);};
	this.meshes = [];

	if(this.options.mode == "solid"){
	    var xy_plane = new THREE.Mesh(geometry, material);
	    var xz_plane = new THREE.Mesh(geometry, material);
	    var yz_plane = new THREE.Mesh(geometry, material);

	    xz_plane.rotateOnAxis(newV(1,0,0), Math.PI/2);
	    xz_plane.translateOnAxis(newV(0,0,-1), -10);

	    yz_plane.rotateOnAxis(newV(0,1,0), Math.PI/2);
	    yz_plane.translateOnAxis(newV(0,0,-1), -10);

	    xy_plane.translateOnAxis(newV(0,0,1), -10);

	    this.meshes.push(xy_plane);
	    this.meshes.push(xz_plane);
	    this.meshes.push(yz_plane);
	}else if(this.options.mode == "wireframe"){
	    var coordinates = [
		[[-10, 10, -10], [-10, -10, -10],[10,-10,-10]],
		[[-10, 10, 10], [-10, -10, 10], [10,-10,10],[10,10,10], [-10, 10, 10]],
		[[10, -10, -10], [10, -10, 10]],
		[[-10, 10, -10], [-10, 10, 10]],
		[[-10, -10, -10], [-10, -10, 10]],
		[[10, -10, -10], [10, 10, -10]]
	    ];
	    var meshes = this.meshes;
	}
	
	this.scales = {};
	this.scales.x = d3.scale.linear().domain([ranges.x.max, ranges.x.min]).range([-10, 10]);
	this.scales.y = d3.scale.linear().domain([ranges.y.max, ranges.y.min]).range([10, -10]);
	this.scales.z = d3.scale.linear().domain([ranges.z.max, ranges.z.min]).range([10,-10]);

	// generate axis
	var x_scale = d3.scale.linear().domain([ranges.x.max, ranges.x.min]).range([20, 0]);
	var y_scale = d3.scale.linear().domain([ranges.y.max, ranges.y.min]).range([20, 0]);
	var z_scale = d3.scale.linear().domain([ranges.z.max, ranges.z.min]).range([20,0]);
	this.meshes = this.meshes.concat(generateAxisAndLabels(this.options.axis_labels.x, newV(10,10,-10),newV(-10,10,-10),newV(0,1,0),x_scale));
	this.meshes = this.meshes.concat(generateAxisAndLabels(this.options.axis_labels.y, newV(-10,-10,-10),newV(-10,10,-10),newV(-1,0,0),y_scale));
	this.meshes = this.meshes.concat(generateAxisAndLabels(this.options.axis_labels.z, newV(10,10,-10),newV(10,10,10),newV(0,1,0),z_scale));

	// generate grids
	if(this.options.grid){
	    this.meshes.push(generateGrid([-10,10],[-10,10],[-10,-10],2));//x-y
	    this.meshes.push(generateGrid([-10,10],[-10,-10],[-10,10],2));//x-z
	    this.meshes.push(generateGrid([10,10],[-10,10],[-10,10],2));//y-z
	}

	return this;
    }

    var generateLabel = function(text, position){
	    var canvas = document.createElement('canvas');
	    canvas.width = 100;
	    canvas.height = 100;
	    var context = canvas.getContext('2d');
	    context.fillStyle = "rgb(0, 0, 0)";
	    context.font = "60px sans-serif";
	    var text_width = context.measureText(text).width;
	    context.fillText(text, (100-text_width)/2, 80);
	    var texture = new THREE.Texture(canvas);
	    texture.flipY = true;
	    texture.needsUpdate = true;
	    var material = new THREE.SpriteMaterial({
	        map: texture,
	        transparent: true,
	        useScreenCoordinates: false
	    });
	    var sprite = new THREE.Sprite(material);
	    sprite.scale.set(1.5,1.5);
	    sprite.position.set.apply(sprite.position, position.toArray());
	    return sprite;
    };

    var generateAxisAndLabels = function(axis_label, axis_start, axis_end, nv_tick, scale){
	    var meshes = [];
	    var geometry = new THREE.Geometry();
	    var nv_start2end = (new THREE.Vector3).subVectors(axis_end, axis_start).normalize();
	
	    geometry.vertices.push(axis_start);
	    geometry.vertices.push(axis_end);
 
	    var label_position = (new THREE.Vector3).addVectors(axis_end, axis_start).divideScalar(2);
	    label_position.add(nv_tick.clone().multiplyScalar(3));
	    meshes.push(generateLabel(axis_label, label_position));

	    // generate d3.js axis
	    var svg = d3.select("body")
	            .append("svg")
	            .style("width", "500")
	            .style("height", "500")
	            .style("display", "none");
	    var ticks = svg.append("g")
	            .call(d3.svg.axis()
		              .scale(scale)
		              .orient("left")
		              .ticks(5))
	            .selectAll(".tick");

	    // parse svg axis, and generate ticks and labels mimicing svg's.
	    var tick_values = [];
	    for(var i=0; i<ticks[0].length; i++){
	        // generate tick line
	        var nattr = ticks[0][i].getAttribute("transform");
	        var valueArr = /translate\(((?:-|\d|.)+),((?:-|\d|.)+)\)/g.exec(nattr);
	        var tick_center = (new THREE.Vector3).addVectors(axis_start, nv_start2end.clone().multiplyScalar(valueArr[2]));
	        var tick_start = (new THREE.Vector3).addVectors(tick_center, nv_tick.clone().multiplyScalar(0.3));
	        var tick_end = (new THREE.Vector3).addVectors(tick_center, nv_tick.clone().multiplyScalar(-0.3));
	        geometry.vertices.push(tick_start);
	        geometry.vertices.push(tick_end);

	        //generate labels
	        var text = ticks[0][i].children[1].childNodes[0].nodeValue;
	        var label_center = (new THREE.Vector3).addVectors(tick_center, nv_tick.clone().multiplyScalar(1.0));
	        var label = generateLabel(text, label_center);
	        meshes.push(label);
	    }

	    svg.remove();

	    var material = new THREE.LineBasicMaterial( { color: 0x444444, opacity: 0.2 } );
	    var line = new THREE.Line(geometry, material, THREE.LinePieces);
	    meshes.push(line);
	    return meshes;
    };

    var generateGrid = function(x_range, y_range, z_range, interval){
	    var geometry = new THREE.Geometry();

	    if(x_range[0]!=x_range[1])for(var x=x_range[0];x<=x_range[1];x+=interval){
	        geometry.vertices.push(new THREE.Vector3(x,y_range[0],z_range[0]));
	        geometry.vertices.push(new THREE.Vector3(x,y_range[1],z_range[1]));
	    }
	    if(y_range[0]!=y_range[1])for(var y=y_range[0];y<=y_range[1];y+=interval){
	        geometry.vertices.push(new THREE.Vector3(x_range[0],y,z_range[0]));
	        geometry.vertices.push(new THREE.Vector3(x_range[1],y,z_range[1]));
	    }
	    if(z_range[0]!=z_range[1])for(var z=z_range[0];z<=z_range[1];z+=interval){
	        geometry.vertices.push(new THREE.Vector3(x_range[0],y_range[0],z));
	        geometry.vertices.push(new THREE.Vector3(x_range[1],y_range[1],z));
	    }
	    var material = new THREE.LineBasicMaterial( { color: 0x444444, opacity: 0.2 } );
	    var line = new THREE.Line(geometry, material, THREE.LinePieces);
	    return line;
    };

    Space.prototype.getScales= function(){
	    return this.scales;
    };

    Space.prototype.getMeshes = function(){
	    return this.meshes;
    };

    return Space;
});

define('utils/database',[],function(){
    var DataBase = {lists:{}};

    // data -> [{t: 0, data: [{x:1, y:2, z:3}, ..., {x:10, y:2, z:5}]}, {t: 100, data: []}, ..., {}]
    // this.lists -> {0:[], 1:[], ..., 100:[]}
    DataBase.add = function(name, data, data_label, seek_label, init){
	    var raw = {}, range=[Infinity,-Infinity];
	    data.forEach(function(row){
	        var val = row[seek_label];
	        raw[row[seek_label]] = row[data_label];
	        range[0] = (val > range[0] ? range[0] : val);
	        range[1] = (val < range[1] ? range[1] : val);
	    });
	    this.lists[name] = {data:raw, seek:init};
	    this.range = range;
    };

    DataBase.seek = function(name, seek){
	    for(var n in this.lists){
	        this.lists[n].seek = seek;
	    }
	    //this.lists[name].seek = seek;
    };

    DataBase.getRange = function(){
	    return this.range;
    };

    DataBase.find = function(name){
	    var seek = this.lists[name].seek;
	    return this.lists[name].data[seek];
    };

    return DataBase;
});

define('components/player',[
    "utils/utils",
    "utils/database"
], function(Utils, DataBase){
    function Player(element, stage, options){
	    this.options = {
	        
	    };

	    if(arguments.length > 1){
	        Utils.merge(this.options, options);
	    };

	    this.model = element;
	    this.stage = stage;
    };
    
    Player.prototype.render = function(){
	    var range = DataBase.getRange();
	    var model = this.model.append("div")
		        .style("height", 27)
		        .style("width", 500)
		        .style("background-color", "#fff");

	    var thisObj = this;

	    model.append("button")
	        .attr("title", "play")
	        .style("float", "left")
	        .text("\u25ba")
	        .on("click", function(){
		        console.log("huga");
                thisObj.start();
	        });

	    var form = model.append("form")
	            .style("height", 30)
	            .style("width", 500);

	    form.append("input")
	        .attr("type", "range")
	        .attr("name", "range")
            .attr("class", "range")
	        .attr("max", range[1])
	        .attr("min", range[0])
            .attr("step", 1)
	        .attr("value", range[0])
	        .style("width", 350)
	        .style("float", "left")
	        .on("change", function(){
		        thisObj.update(this.value);
	        });

	    form.append("input")
	        .attr("type", "text")
	        .style("width", 30)
	        .style("float", "left")
	        .attr("value", range[0])
	        .attr("class", "input_label");
	    
	    form.append("div")
	        .style("color", "#fff")
	        .append("p").style("line-height", 25)
	        .text(range[1]);

        this.form = form;
    };

    Player.prototype.start = function(){
        var target_player = this;
        var timer = window.setInterval("timer_func()", 400);

        window["timer_func"] = function(){
            var val, step, max;
            target_player
                .form
                .select(".range")
                .each(function(){
                    var selector = d3.select(this);
                    val = parseInt(this.value);
                    step = parseInt(selector.attr("step"));
                    max = parseInt(selector.attr("max"));
                });

            if(val+step <= max){
                console.log("called!");
                target_player.form
                    .select(".range")
                    .each(function(selector){
                        this.value = val + step;
                    });
                target_player.update(val + step);
            }else{
                window.clearInterval(timer);
            }
        };
    };

    Player.prototype.update = function(val){
	    DataBase.seek("", val);
	    this.model.select(".input_label").attr("value", val);
	    this.stage.clear();
	    this.stage.update();
    };

    return Player;
});

define('components/menu',[
    "utils/utils"
], function(Utils){
    function Menu(selection, options){
	this.options = {
	    filename: "plot"
	};

	if(arguments.length > 1){
	    Utils.merge(this.options, options);
	};

	this.selection = selection;
    }

    Menu.prototype.begin = function(){
	var filename = this.options.filename;

	function removeMenu(){
	    d3.selectAll(".download_menu").remove();
	}

	function createMenu(pos, canvas){
	    removeMenu();

	    var ul = d3.select("body")
		    .append("ul")
		    .on("click", removeMenu);

	    ul.style({
		"list-style-type": "none",
		position: "absolute",
		left: pos[0] + "px",
		top: pos[1] + "px",
		background: "#f3f3f3",
		border: "1px solid #fff",
		padding: 10,
		margin: 0
	    }).attr("class", "download_menu");

	    ul.append("li")
		.append("a")
		.text("save as png")
		.attr({
		    download: filename + ".png",
		    href: canvas.toDataURL("image/png")
		});

	    ul.append("li")
		.append("a")
		.text("save as jpeg")
		.attr({
		    download: filename + ".jpg",
		    href: canvas.toDataURL("image/jpeg")
		});

	    ul.selectAll("a")
		.style({
		    display: "block",
		    "text-decoration": "none",
		    "text-align": "left",
		    "line-style": "none",
		    "color": "#333",
		    "font-size" : "13px",
		    "line-height" : "17px"
		});

	    ul.selectAll("li")
		.style({
		    "margin": 0
		});
	}

	this.selection.select("canvas")
	    .on("contextmenu", function(){
		var pos = d3.mouse(document.body);
		createMenu(pos, this, filename);
		d3.event.preventDefault();
	    })
	    .on("click", removeMenu);
    };

    return Menu;
});

define('utils/range',[],function(){
    function Range(arg0, arg1){
	if(arguments.length > 1){
	    this.max = arg0;
	    this.min = arg1;
	}else{
	    this.max  = arg0[1];
	    this.min  = arg0[0];
	}
    }

    Range.prototype.divide = function(num){
	var arr = new Array();
	var interval = Math.ceil((this.max-this.min)/(num-1));
	for(var i=0;i<num;i++){
	    arr.push(this.min + interval*i);
	}
	return arr;
    };

    Range.expand = function(range1, range2){
	return new Range(Math.max(range1.max, range2.max), Math.min(range1.min, range2.min));
    };

    return Range;
});

define('components/stage',[
    "components/world",
    "components/space",
    "components/player",
    "components/menu",
    "utils/utils",
    "utils/range"
], function(World, Space, Player, Menu, Utils, Range){
    function Stage(element, options){
	this.options = {
	    width:700,
	    height:530,
	    world_width:900,
	    world_height:800,
	    axis_labels: {x:"X", y:"Y", z:"Z"},
	    bg_color:0x1e2429,
	    player: false,
	    space_mode: 'wireframe',
	    range:{x:[0,0], y:[0,0], z:[0,0]},
	    autorange:true,
	    grid: true,
	    perspective: true,
	    orbit: false,
	    save_image: false
	};

	if(arguments.length > 1){
	    Utils.merge(this.options, options);
	};
	
	var selection = d3.select(element);
	selection.style("width",String(this.options.width));

	this.world_space = selection.append("div")
	    .style({
		"float":"left",
		"width":String(this.options.world_width),
		"height": String(this.options.world_height),
		"save_image": this.options.save_image
	    });

	this.legend_space = selection.append("div")
	    .style({
		"float":"left",
		"width":String(this.options.width - this.options.world_width),
		"height":String(this.options.height)
	    });

	if(this.options.player){
	    var player_space = selection.append("div")
		    .style("width",String(this.options.width))
		    .style("height",String(this.options.height - this.options.world_height));

	    this.player = new Player(player_space, this);
	}

	if(this.options.save_image){
	    this.menu = new Menu(this.world_space);
	}

	this.charts = [];

	this.world = new World({
	    width:this.options.world_width,
	    height:this.options.world_height,
	    bg_color:this.options.bg_color,
	    perspective: this.options.perspective,
        orbit: this.options.orbit
	});

	this.data_ranges = {
            x:new Range(this.options.range.x[0], this.options.range.x[1]),
            y:new Range(this.options.range.y[0], this.options.range.y[1]),
            z:new Range(this.options.range.z[0], this.options.range.z[1])
        };

	return this;
    }

    Stage.prototype.add = function(chart){
        if(this.options.autorange){
            var ranges = chart.getDataRanges();
            var thisObj=this;
            ['x', 'y', 'z'].forEach(function(i){
                thisObj.data_ranges[i] = Range.expand(thisObj.data_ranges[i], ranges[i]);
            });
        }
	this.charts.push(chart);
    };

    Stage.prototype.render = function(){
	this.space = new Space(this.data_ranges, {
	    axis_labels:this.options.axis_labels,
	    mode: this.options.space_mode,
	    grid: this.options.grid
	});
	this.world.addMesh(this.space.getMeshes());
        for(var i=0;i<this.charts.length;i++){
            var chart=this.charts[i];
            chart.generateMesh(this.space.getScales(), this);
	    this.world.addMesh(chart.getMesh());
            if(chart.hasLegend()){
		var legend = chart.getLegend();
		this.legend_space[0][0].appendChild(legend[0][0]);
	    }
        }

	if(this.options.player){
	    this.player.render();
	}

	this.world.begin(this.world_space);
	if(this.options.save_image)this.menu.begin();
    };

    Stage.prototype.dispose = function(){
	this.clear();
	this.world.renderer.clear();
    };

    Stage.prototype.clear = function(){
        for(var i=0;i<this.charts.length;i++){
            var chart=this.charts[i];
	    this.world.removeMesh(chart.getMesh());
	}
    };

    Stage.prototype.update = function(){
        for(var i=0;i<this.charts.length;i++){
            var chart=this.charts[i];
            chart.generateMesh(this.space.getScales(), this);
	    this.world.addMesh(chart.getMesh());
        }
    };

    return Stage;
});

define('components/legends',[],function(){

    function generateContinuousLegend(range, color){
	var scale = d3.scale.linear().domain([range.max, range.min]).range([0,200]);

	var div = d3.select(document.createElement("div"))
	    .style("padding", "5px")
	    .style("float", "left")
	    .style("width","100")
	    .style("height","auto");
	
	var svg = div.append("svg")
	    .style("height","100%") // fixed for Mozilla Firefox Bug 736431
	    .style("width", "100px");

	var gradient = svg.append("svg:defs")
	    .append("svg:linearGradient")
	    .attr("id", "gradient")
	    .attr("x1", "0%")
	    .attr("x2", "0%")
	    .attr("y1", "100%")
	    .attr("y2", "0%");

	for(var i=0; i<color.length; i++){
	    gradient.append("svg:stop")
		.attr("offset", (100/(color.length-1))*i + "%")
		.attr("stop-color", color[i]);
	}

	var group = svg.append("g");

	group.append("svg:rect")
	    .attr("y",10)
	    .attr("width", "25")
	    .attr("height", "200")
	    .style("fill", "url(#gradient)");
	
	svg.append("g")
	    .attr("width", "100")
	    .attr("height", "200")
	    .attr("class", "axis")
	    .attr("transform", "translate(25,10)")
	    .call(d3.svg.axis()
		  .scale(scale)
		  .orient("right")
		  .ticks(5));

	svg.selectAll(".axis").selectAll("path")
	    .style("fill", "none")
	    .style("stroke", "black")
	    .style("shape-rendering", "crispEdges");

	svg.selectAll(".axis").selectAll("line")
	    .style("fill", "none")
	    .style("stroke", "black")
	    .style("shape-rendering", "crispEdges");

	svg.selectAll(".axis").selectAll("text")
	    .style("font-family", "san-serif")
	    .style("font-size", "11px");

	return div;
    };

    function generateDiscreteLegend(name, color, chart){
	var div = d3.select(document.createElement("div"))
	    .style("padding", "4")
	    .style("height","16")
	    .style("width","100%");
	
	var svg = div.append("svg")
	    .style("height","30px") // fixed for Mozilla Firefox Bug 736431
	    .style("width", "100px");
	
	var onclick_func = function(event){
	    var element = event.target;
	    if(element.getAttribute("fill-opacity")=="0.0"){
		element.setAttribute("fill-opacity","1.0");
		element.chart.appear();
	    }else{
		element.setAttribute("fill-opacity","0.0");
		element.chart.disappear();
	    }
	};

	var circle = svg.append("circle")
	    .attr("cx","8")
	    .attr("cy","8")
	    .attr("r","6")
	    .attr("stroke",color)
	    .attr("stroke-width","2")
	    .attr("fill",color)
	    .style("cursor","pointer");

	circle[0][0].chart = chart;

	circle[0][0].onclick = onclick_func;

	svg.append("text")
	    .attr("x","18")
	    .attr("y","12")
	    .attr("font-size","12")
	    .text(name);

	return div;
    }

    var Legends = {
	generateContinuousLegend:generateContinuousLegend,
	generateDiscreteLegend:generateDiscreteLegend
    };

    return Legends;
});

define('utils/datasets',[
    "utils/range",
    "utils/database"
],function(Range, DataBase){

    /*
     MatrixDataset:

     *** arguments ***
     data: [object] 3 nested array like ones generated by numpy.meshgrid.
     e.g. {
        x: [
           [1,2,3],
           [2,4,5],
        ],
        y: [
           [1,2,3],
           [2,4,5],
        ],
        z: [
           [1,2,3],
           [2,4,5],
        ],
     }

     *** properties ***
     ranges: the range of each column
     raw: given data
     */
    function MatrixDataset(data){
	var ranges = {};
	if(typeof data == "string"){
	    this.raw = DataBase.find(data);
	}else{
	    this.raw = data;
	}
	for(var i in data){
	    ranges[i] = new Range(
		d3.max(this.raw[i], function(d){return Math.max.apply(null,d);}),
		d3.min(this.raw[i], function(d){return Math.min.apply(null,d);})
	    );
	}
	this.ranges = ranges;
	this.raw = data;
	return this;
    }

    /*
     ArrayDataset: 
     *** arguments ***
     data: [object] 2 nested array
     e.g. {
        x: [1,2,3,4,...,10], // x
        y: [2,3,4,5,...,11], // y
        z: [3,4,5,6,...,12]  // z
     }

     *** properties ***
     ranges: the range of each column
     raw: given data
     */
    function ArrayDataset(data){
	this.ranges = {};
	if(typeof data == "string"){
	    this.raw = DataBase.find(data);
        if(_.isUndefined(this.raw))
            this.raw = {x: [], y: [], z:[]};
	}else{
	    this.raw = data;
	}
	for(var i in this.raw){
	    this.ranges[i] = new Range(
		d3.max(this.raw[i]),
		d3.min(this.raw[i])
	    );
	}
    }

    /*
     CompressedDataset: compressed data for volume rendering

     *** arguments ***
     data: [string] base64 encoded image
     e.g.
     'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAANElEQVR4nO3BAQ0AAADCoPd
     PbQ43oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAfgx1lAABqFDyOQAAAABJRU5ErkJggolQTkcNChoKAAAADUlIR
     FIAAABkAAAAZAgCAAAA/4ACAwAAADRJREFUeJztwQENAAAAwqD3T20ON6AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
     AAH4MdZQAAahQ8jkAAAAASUVORK5CYII='

     *** properties ***
     ranges: the range of each column
     */
    function CompressedDataset(data){
	this.raw = data;
	this.ranges = {
	    x: [0, 1],
	    y: [0, 1],
	    z: [0, 1]
	};
	return this;
    }

    var Datasets = {
	Matrix:MatrixDataset,
	Array:ArrayDataset,
	Compressed: CompressedDataset
    };

    return Datasets;
});

// This product includes color specifications and designs developed by Cynthia Brewer (http://colorbrewer.org/).
// JavaScript specs as packaged in the D3 library (d3js.org). Please see license at http://colorbrewer.org/export/LICENSE.txt

define('utils/colorbrewer',[],function(){

    var colorbrewer = {YlGn: {
	3: ["#f7fcb9","#addd8e","#31a354"],
	4: ["#ffffcc","#c2e699","#78c679","#238443"],
	5: ["#ffffcc","#c2e699","#78c679","#31a354","#006837"],
	6: ["#ffffcc","#d9f0a3","#addd8e","#78c679","#31a354","#006837"],
	7: ["#ffffcc","#d9f0a3","#addd8e","#78c679","#41ab5d","#238443","#005a32"],
	8: ["#ffffe5","#f7fcb9","#d9f0a3","#addd8e","#78c679","#41ab5d","#238443","#005a32"],
	9: ["#ffffe5","#f7fcb9","#d9f0a3","#addd8e","#78c679","#41ab5d","#238443","#006837","#004529"]
    },YlGnBu: {
	3: ["#edf8b1","#7fcdbb","#2c7fb8"],
	4: ["#ffffcc","#a1dab4","#41b6c4","#225ea8"],
	5: ["#ffffcc","#a1dab4","#41b6c4","#2c7fb8","#253494"],
	6: ["#ffffcc","#c7e9b4","#7fcdbb","#41b6c4","#2c7fb8","#253494"],
	7: ["#ffffcc","#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8","#0c2c84"],
	8: ["#ffffd9","#edf8b1","#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8","#0c2c84"],
	9: ["#ffffd9","#edf8b1","#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8","#253494","#081d58"]
    },GnBu: {
	3: ["#e0f3db","#a8ddb5","#43a2ca"],
	4: ["#f0f9e8","#bae4bc","#7bccc4","#2b8cbe"],
	5: ["#f0f9e8","#bae4bc","#7bccc4","#43a2ca","#0868ac"],
	6: ["#f0f9e8","#ccebc5","#a8ddb5","#7bccc4","#43a2ca","#0868ac"],
	7: ["#f0f9e8","#ccebc5","#a8ddb5","#7bccc4","#4eb3d3","#2b8cbe","#08589e"],
	8: ["#f7fcf0","#e0f3db","#ccebc5","#a8ddb5","#7bccc4","#4eb3d3","#2b8cbe","#08589e"],
	9: ["#f7fcf0","#e0f3db","#ccebc5","#a8ddb5","#7bccc4","#4eb3d3","#2b8cbe","#0868ac","#084081"]
    },BuGn: {
	3: ["#e5f5f9","#99d8c9","#2ca25f"],
	4: ["#edf8fb","#b2e2e2","#66c2a4","#238b45"],
	5: ["#edf8fb","#b2e2e2","#66c2a4","#2ca25f","#006d2c"],
	6: ["#edf8fb","#ccece6","#99d8c9","#66c2a4","#2ca25f","#006d2c"],
	7: ["#edf8fb","#ccece6","#99d8c9","#66c2a4","#41ae76","#238b45","#005824"],
	8: ["#f7fcfd","#e5f5f9","#ccece6","#99d8c9","#66c2a4","#41ae76","#238b45","#005824"],
	9: ["#f7fcfd","#e5f5f9","#ccece6","#99d8c9","#66c2a4","#41ae76","#238b45","#006d2c","#00441b"]
    },PuBuGn: {
	3: ["#ece2f0","#a6bddb","#1c9099"],
	4: ["#f6eff7","#bdc9e1","#67a9cf","#02818a"],
	5: ["#f6eff7","#bdc9e1","#67a9cf","#1c9099","#016c59"],
	6: ["#f6eff7","#d0d1e6","#a6bddb","#67a9cf","#1c9099","#016c59"],
	7: ["#f6eff7","#d0d1e6","#a6bddb","#67a9cf","#3690c0","#02818a","#016450"],
	8: ["#fff7fb","#ece2f0","#d0d1e6","#a6bddb","#67a9cf","#3690c0","#02818a","#016450"],
	9: ["#fff7fb","#ece2f0","#d0d1e6","#a6bddb","#67a9cf","#3690c0","#02818a","#016c59","#014636"]
    },PuBu: {
	3: ["#ece7f2","#a6bddb","#2b8cbe"],
	4: ["#f1eef6","#bdc9e1","#74a9cf","#0570b0"],
	5: ["#f1eef6","#bdc9e1","#74a9cf","#2b8cbe","#045a8d"],
	6: ["#f1eef6","#d0d1e6","#a6bddb","#74a9cf","#2b8cbe","#045a8d"],
	7: ["#f1eef6","#d0d1e6","#a6bddb","#74a9cf","#3690c0","#0570b0","#034e7b"],
	8: ["#fff7fb","#ece7f2","#d0d1e6","#a6bddb","#74a9cf","#3690c0","#0570b0","#034e7b"],
	9: ["#fff7fb","#ece7f2","#d0d1e6","#a6bddb","#74a9cf","#3690c0","#0570b0","#045a8d","#023858"]
    },BuPu: {
	3: ["#e0ecf4","#9ebcda","#8856a7"],
	4: ["#edf8fb","#b3cde3","#8c96c6","#88419d"],
	5: ["#edf8fb","#b3cde3","#8c96c6","#8856a7","#810f7c"],
	6: ["#edf8fb","#bfd3e6","#9ebcda","#8c96c6","#8856a7","#810f7c"],
	7: ["#edf8fb","#bfd3e6","#9ebcda","#8c96c6","#8c6bb1","#88419d","#6e016b"],
	8: ["#f7fcfd","#e0ecf4","#bfd3e6","#9ebcda","#8c96c6","#8c6bb1","#88419d","#6e016b"],
	9: ["#f7fcfd","#e0ecf4","#bfd3e6","#9ebcda","#8c96c6","#8c6bb1","#88419d","#810f7c","#4d004b"]
    },RdPu: {
	3: ["#fde0dd","#fa9fb5","#c51b8a"],
	4: ["#feebe2","#fbb4b9","#f768a1","#ae017e"],
	5: ["#feebe2","#fbb4b9","#f768a1","#c51b8a","#7a0177"],
	6: ["#feebe2","#fcc5c0","#fa9fb5","#f768a1","#c51b8a","#7a0177"],
	7: ["#feebe2","#fcc5c0","#fa9fb5","#f768a1","#dd3497","#ae017e","#7a0177"],
	8: ["#fff7f3","#fde0dd","#fcc5c0","#fa9fb5","#f768a1","#dd3497","#ae017e","#7a0177"],
	9: ["#fff7f3","#fde0dd","#fcc5c0","#fa9fb5","#f768a1","#dd3497","#ae017e","#7a0177","#49006a"]
    },PuRd: {
	3: ["#e7e1ef","#c994c7","#dd1c77"],
	4: ["#f1eef6","#d7b5d8","#df65b0","#ce1256"],
	5: ["#f1eef6","#d7b5d8","#df65b0","#dd1c77","#980043"],
	6: ["#f1eef6","#d4b9da","#c994c7","#df65b0","#dd1c77","#980043"],
	7: ["#f1eef6","#d4b9da","#c994c7","#df65b0","#e7298a","#ce1256","#91003f"],
	8: ["#f7f4f9","#e7e1ef","#d4b9da","#c994c7","#df65b0","#e7298a","#ce1256","#91003f"],
	9: ["#f7f4f9","#e7e1ef","#d4b9da","#c994c7","#df65b0","#e7298a","#ce1256","#980043","#67001f"]
    },OrRd: {
	3: ["#fee8c8","#fdbb84","#e34a33"],
	4: ["#fef0d9","#fdcc8a","#fc8d59","#d7301f"],
	5: ["#fef0d9","#fdcc8a","#fc8d59","#e34a33","#b30000"],
	6: ["#fef0d9","#fdd49e","#fdbb84","#fc8d59","#e34a33","#b30000"],
	7: ["#fef0d9","#fdd49e","#fdbb84","#fc8d59","#ef6548","#d7301f","#990000"],
	8: ["#fff7ec","#fee8c8","#fdd49e","#fdbb84","#fc8d59","#ef6548","#d7301f","#990000"],
	9: ["#fff7ec","#fee8c8","#fdd49e","#fdbb84","#fc8d59","#ef6548","#d7301f","#b30000","#7f0000"]
    },YlOrRd: {
	3: ["#ffeda0","#feb24c","#f03b20"],
	4: ["#ffffb2","#fecc5c","#fd8d3c","#e31a1c"],
	5: ["#ffffb2","#fecc5c","#fd8d3c","#f03b20","#bd0026"],
	6: ["#ffffb2","#fed976","#feb24c","#fd8d3c","#f03b20","#bd0026"],
	7: ["#ffffb2","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#b10026"],
	8: ["#ffffcc","#ffeda0","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#b10026"],
	9: ["#ffffcc","#ffeda0","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#bd0026","#800026"]
    },YlOrBr: {
	3: ["#fff7bc","#fec44f","#d95f0e"],
	4: ["#ffffd4","#fed98e","#fe9929","#cc4c02"],
	5: ["#ffffd4","#fed98e","#fe9929","#d95f0e","#993404"],
	6: ["#ffffd4","#fee391","#fec44f","#fe9929","#d95f0e","#993404"],
	7: ["#ffffd4","#fee391","#fec44f","#fe9929","#ec7014","#cc4c02","#8c2d04"],
	8: ["#ffffe5","#fff7bc","#fee391","#fec44f","#fe9929","#ec7014","#cc4c02","#8c2d04"],
	9: ["#ffffe5","#fff7bc","#fee391","#fec44f","#fe9929","#ec7014","#cc4c02","#993404","#662506"]
    },Purples: {
	3: ["#efedf5","#bcbddc","#756bb1"],
	4: ["#f2f0f7","#cbc9e2","#9e9ac8","#6a51a3"],
	5: ["#f2f0f7","#cbc9e2","#9e9ac8","#756bb1","#54278f"],
	6: ["#f2f0f7","#dadaeb","#bcbddc","#9e9ac8","#756bb1","#54278f"],
	7: ["#f2f0f7","#dadaeb","#bcbddc","#9e9ac8","#807dba","#6a51a3","#4a1486"],
	8: ["#fcfbfd","#efedf5","#dadaeb","#bcbddc","#9e9ac8","#807dba","#6a51a3","#4a1486"],
	9: ["#fcfbfd","#efedf5","#dadaeb","#bcbddc","#9e9ac8","#807dba","#6a51a3","#54278f","#3f007d"]
    },Blues: {
	3: ["#deebf7","#9ecae1","#3182bd"],
	4: ["#eff3ff","#bdd7e7","#6baed6","#2171b5"],
	5: ["#eff3ff","#bdd7e7","#6baed6","#3182bd","#08519c"],
	6: ["#eff3ff","#c6dbef","#9ecae1","#6baed6","#3182bd","#08519c"],
	7: ["#eff3ff","#c6dbef","#9ecae1","#6baed6","#4292c6","#2171b5","#084594"],
	8: ["#f7fbff","#deebf7","#c6dbef","#9ecae1","#6baed6","#4292c6","#2171b5","#084594"],
	9: ["#f7fbff","#deebf7","#c6dbef","#9ecae1","#6baed6","#4292c6","#2171b5","#08519c","#08306b"]
    },Greens: {
	3: ["#e5f5e0","#a1d99b","#31a354"],
	4: ["#edf8e9","#bae4b3","#74c476","#238b45"],
	5: ["#edf8e9","#bae4b3","#74c476","#31a354","#006d2c"],
	6: ["#edf8e9","#c7e9c0","#a1d99b","#74c476","#31a354","#006d2c"],
	7: ["#edf8e9","#c7e9c0","#a1d99b","#74c476","#41ab5d","#238b45","#005a32"],
	8: ["#f7fcf5","#e5f5e0","#c7e9c0","#a1d99b","#74c476","#41ab5d","#238b45","#005a32"],
	9: ["#f7fcf5","#e5f5e0","#c7e9c0","#a1d99b","#74c476","#41ab5d","#238b45","#006d2c","#00441b"]
    },Oranges: {
	3: ["#fee6ce","#fdae6b","#e6550d"],
	4: ["#feedde","#fdbe85","#fd8d3c","#d94701"],
	5: ["#feedde","#fdbe85","#fd8d3c","#e6550d","#a63603"],
	6: ["#feedde","#fdd0a2","#fdae6b","#fd8d3c","#e6550d","#a63603"],
	7: ["#feedde","#fdd0a2","#fdae6b","#fd8d3c","#f16913","#d94801","#8c2d04"],
	8: ["#fff5eb","#fee6ce","#fdd0a2","#fdae6b","#fd8d3c","#f16913","#d94801","#8c2d04"],
	9: ["#fff5eb","#fee6ce","#fdd0a2","#fdae6b","#fd8d3c","#f16913","#d94801","#a63603","#7f2704"]
    },Reds: {
	3: ["#fee0d2","#fc9272","#de2d26"],
	4: ["#fee5d9","#fcae91","#fb6a4a","#cb181d"],
	5: ["#fee5d9","#fcae91","#fb6a4a","#de2d26","#a50f15"],
	6: ["#fee5d9","#fcbba1","#fc9272","#fb6a4a","#de2d26","#a50f15"],
	7: ["#fee5d9","#fcbba1","#fc9272","#fb6a4a","#ef3b2c","#cb181d","#99000d"],
	8: ["#fff5f0","#fee0d2","#fcbba1","#fc9272","#fb6a4a","#ef3b2c","#cb181d","#99000d"],
	9: ["#fff5f0","#fee0d2","#fcbba1","#fc9272","#fb6a4a","#ef3b2c","#cb181d","#a50f15","#67000d"]
    },Greys: {
	3: ["#f0f0f0","#bdbdbd","#636363"],
	4: ["#f7f7f7","#cccccc","#969696","#525252"],
	5: ["#f7f7f7","#cccccc","#969696","#636363","#252525"],
	6: ["#f7f7f7","#d9d9d9","#bdbdbd","#969696","#636363","#252525"],
	7: ["#f7f7f7","#d9d9d9","#bdbdbd","#969696","#737373","#525252","#252525"],
	8: ["#ffffff","#f0f0f0","#d9d9d9","#bdbdbd","#969696","#737373","#525252","#252525"],
	9: ["#ffffff","#f0f0f0","#d9d9d9","#bdbdbd","#969696","#737373","#525252","#252525","#000000"]
    },PuOr: {
	3: ["#f1a340","#f7f7f7","#998ec3"],
	4: ["#e66101","#fdb863","#b2abd2","#5e3c99"],
	5: ["#e66101","#fdb863","#f7f7f7","#b2abd2","#5e3c99"],
	6: ["#b35806","#f1a340","#fee0b6","#d8daeb","#998ec3","#542788"],
	7: ["#b35806","#f1a340","#fee0b6","#f7f7f7","#d8daeb","#998ec3","#542788"],
	8: ["#b35806","#e08214","#fdb863","#fee0b6","#d8daeb","#b2abd2","#8073ac","#542788"],
	9: ["#b35806","#e08214","#fdb863","#fee0b6","#f7f7f7","#d8daeb","#b2abd2","#8073ac","#542788"],
	10: ["#7f3b08","#b35806","#e08214","#fdb863","#fee0b6","#d8daeb","#b2abd2","#8073ac","#542788","#2d004b"],
	11: ["#7f3b08","#b35806","#e08214","#fdb863","#fee0b6","#f7f7f7","#d8daeb","#b2abd2","#8073ac","#542788","#2d004b"]
    },BrBG: {
	3: ["#d8b365","#f5f5f5","#5ab4ac"],
	4: ["#a6611a","#dfc27d","#80cdc1","#018571"],
	5: ["#a6611a","#dfc27d","#f5f5f5","#80cdc1","#018571"],
	6: ["#8c510a","#d8b365","#f6e8c3","#c7eae5","#5ab4ac","#01665e"],
	7: ["#8c510a","#d8b365","#f6e8c3","#f5f5f5","#c7eae5","#5ab4ac","#01665e"],
	8: ["#8c510a","#bf812d","#dfc27d","#f6e8c3","#c7eae5","#80cdc1","#35978f","#01665e"],
	9: ["#8c510a","#bf812d","#dfc27d","#f6e8c3","#f5f5f5","#c7eae5","#80cdc1","#35978f","#01665e"],
	10: ["#543005","#8c510a","#bf812d","#dfc27d","#f6e8c3","#c7eae5","#80cdc1","#35978f","#01665e","#003c30"],
	11: ["#543005","#8c510a","#bf812d","#dfc27d","#f6e8c3","#f5f5f5","#c7eae5","#80cdc1","#35978f","#01665e","#003c30"]
    },PRGn: {
	3: ["#af8dc3","#f7f7f7","#7fbf7b"],
	4: ["#7b3294","#c2a5cf","#a6dba0","#008837"],
	5: ["#7b3294","#c2a5cf","#f7f7f7","#a6dba0","#008837"],
	6: ["#762a83","#af8dc3","#e7d4e8","#d9f0d3","#7fbf7b","#1b7837"],
	7: ["#762a83","#af8dc3","#e7d4e8","#f7f7f7","#d9f0d3","#7fbf7b","#1b7837"],
	8: ["#762a83","#9970ab","#c2a5cf","#e7d4e8","#d9f0d3","#a6dba0","#5aae61","#1b7837"],
	9: ["#762a83","#9970ab","#c2a5cf","#e7d4e8","#f7f7f7","#d9f0d3","#a6dba0","#5aae61","#1b7837"],
	10: ["#40004b","#762a83","#9970ab","#c2a5cf","#e7d4e8","#d9f0d3","#a6dba0","#5aae61","#1b7837","#00441b"],
	11: ["#40004b","#762a83","#9970ab","#c2a5cf","#e7d4e8","#f7f7f7","#d9f0d3","#a6dba0","#5aae61","#1b7837","#00441b"]
    },PiYG: {
	3: ["#e9a3c9","#f7f7f7","#a1d76a"],
	4: ["#d01c8b","#f1b6da","#b8e186","#4dac26"],
	5: ["#d01c8b","#f1b6da","#f7f7f7","#b8e186","#4dac26"],
	6: ["#c51b7d","#e9a3c9","#fde0ef","#e6f5d0","#a1d76a","#4d9221"],
	7: ["#c51b7d","#e9a3c9","#fde0ef","#f7f7f7","#e6f5d0","#a1d76a","#4d9221"],
	8: ["#c51b7d","#de77ae","#f1b6da","#fde0ef","#e6f5d0","#b8e186","#7fbc41","#4d9221"],
	9: ["#c51b7d","#de77ae","#f1b6da","#fde0ef","#f7f7f7","#e6f5d0","#b8e186","#7fbc41","#4d9221"],
	10: ["#8e0152","#c51b7d","#de77ae","#f1b6da","#fde0ef","#e6f5d0","#b8e186","#7fbc41","#4d9221","#276419"],
	11: ["#8e0152","#c51b7d","#de77ae","#f1b6da","#fde0ef","#f7f7f7","#e6f5d0","#b8e186","#7fbc41","#4d9221","#276419"]
    },RdBu: {
	3: ["#ef8a62","#f7f7f7","#67a9cf"],
	4: ["#ca0020","#f4a582","#92c5de","#0571b0"],
	5: ["#ca0020","#f4a582","#f7f7f7","#92c5de","#0571b0"],
	6: ["#b2182b","#ef8a62","#fddbc7","#d1e5f0","#67a9cf","#2166ac"],
	7: ["#b2182b","#ef8a62","#fddbc7","#f7f7f7","#d1e5f0","#67a9cf","#2166ac"],
	8: ["#b2182b","#d6604d","#f4a582","#fddbc7","#d1e5f0","#92c5de","#4393c3","#2166ac"],
	9: ["#b2182b","#d6604d","#f4a582","#fddbc7","#f7f7f7","#d1e5f0","#92c5de","#4393c3","#2166ac"],
	10: ["#67001f","#b2182b","#d6604d","#f4a582","#fddbc7","#d1e5f0","#92c5de","#4393c3","#2166ac","#053061"],
	11: ["#67001f","#b2182b","#d6604d","#f4a582","#fddbc7","#f7f7f7","#d1e5f0","#92c5de","#4393c3","#2166ac","#053061"]
    },RdGy: {
	3: ["#ef8a62","#ffffff","#999999"],
	4: ["#ca0020","#f4a582","#bababa","#404040"],
	5: ["#ca0020","#f4a582","#ffffff","#bababa","#404040"],
	6: ["#b2182b","#ef8a62","#fddbc7","#e0e0e0","#999999","#4d4d4d"],
	7: ["#b2182b","#ef8a62","#fddbc7","#ffffff","#e0e0e0","#999999","#4d4d4d"],
	8: ["#b2182b","#d6604d","#f4a582","#fddbc7","#e0e0e0","#bababa","#878787","#4d4d4d"],
	9: ["#b2182b","#d6604d","#f4a582","#fddbc7","#ffffff","#e0e0e0","#bababa","#878787","#4d4d4d"],
	10: ["#67001f","#b2182b","#d6604d","#f4a582","#fddbc7","#e0e0e0","#bababa","#878787","#4d4d4d","#1a1a1a"],
	11: ["#67001f","#b2182b","#d6604d","#f4a582","#fddbc7","#ffffff","#e0e0e0","#bababa","#878787","#4d4d4d","#1a1a1a"]
    },RdYlBu: {
	3: ["#fc8d59","#ffffbf","#91bfdb"],
	4: ["#d7191c","#fdae61","#abd9e9","#2c7bb6"],
	5: ["#d7191c","#fdae61","#ffffbf","#abd9e9","#2c7bb6"],
	6: ["#d73027","#fc8d59","#fee090","#e0f3f8","#91bfdb","#4575b4"],
	7: ["#d73027","#fc8d59","#fee090","#ffffbf","#e0f3f8","#91bfdb","#4575b4"],
	8: ["#d73027","#f46d43","#fdae61","#fee090","#e0f3f8","#abd9e9","#74add1","#4575b4"],
	9: ["#d73027","#f46d43","#fdae61","#fee090","#ffffbf","#e0f3f8","#abd9e9","#74add1","#4575b4"],
	10: ["#a50026","#d73027","#f46d43","#fdae61","#fee090","#e0f3f8","#abd9e9","#74add1","#4575b4","#313695"],
	11: ["#a50026","#d73027","#f46d43","#fdae61","#fee090","#ffffbf","#e0f3f8","#abd9e9","#74add1","#4575b4","#313695"]
    },Spectral: {
	3: ["#fc8d59","#ffffbf","#99d594"],
	4: ["#d7191c","#fdae61","#abdda4","#2b83ba"],
	5: ["#d7191c","#fdae61","#ffffbf","#abdda4","#2b83ba"],
	6: ["#d53e4f","#fc8d59","#fee08b","#e6f598","#99d594","#3288bd"],
	7: ["#d53e4f","#fc8d59","#fee08b","#ffffbf","#e6f598","#99d594","#3288bd"],
	8: ["#d53e4f","#f46d43","#fdae61","#fee08b","#e6f598","#abdda4","#66c2a5","#3288bd"],
	9: ["#d53e4f","#f46d43","#fdae61","#fee08b","#ffffbf","#e6f598","#abdda4","#66c2a5","#3288bd"],
	10: ["#9e0142","#d53e4f","#f46d43","#fdae61","#fee08b","#e6f598","#abdda4","#66c2a5","#3288bd","#5e4fa2"],
	11: ["#9e0142","#d53e4f","#f46d43","#fdae61","#fee08b","#ffffbf","#e6f598","#abdda4","#66c2a5","#3288bd","#5e4fa2"]
    },RdYlGn: {
	3: ["#fc8d59","#ffffbf","#91cf60"],
	4: ["#d7191c","#fdae61","#a6d96a","#1a9641"],
	5: ["#d7191c","#fdae61","#ffffbf","#a6d96a","#1a9641"],
	6: ["#d73027","#fc8d59","#fee08b","#d9ef8b","#91cf60","#1a9850"],
	7: ["#d73027","#fc8d59","#fee08b","#ffffbf","#d9ef8b","#91cf60","#1a9850"],
	8: ["#d73027","#f46d43","#fdae61","#fee08b","#d9ef8b","#a6d96a","#66bd63","#1a9850"],
	9: ["#d73027","#f46d43","#fdae61","#fee08b","#ffffbf","#d9ef8b","#a6d96a","#66bd63","#1a9850"],
	10: ["#a50026","#d73027","#f46d43","#fdae61","#fee08b","#d9ef8b","#a6d96a","#66bd63","#1a9850","#006837"],
	11: ["#a50026","#d73027","#f46d43","#fdae61","#fee08b","#ffffbf","#d9ef8b","#a6d96a","#66bd63","#1a9850","#006837"]
    },Accent: {
	3: ["#7fc97f","#beaed4","#fdc086"],
	4: ["#7fc97f","#beaed4","#fdc086","#ffff99"],
	5: ["#7fc97f","#beaed4","#fdc086","#ffff99","#386cb0"],
	6: ["#7fc97f","#beaed4","#fdc086","#ffff99","#386cb0","#f0027f"],
	7: ["#7fc97f","#beaed4","#fdc086","#ffff99","#386cb0","#f0027f","#bf5b17"],
	8: ["#7fc97f","#beaed4","#fdc086","#ffff99","#386cb0","#f0027f","#bf5b17","#666666"]
    },Dark2: {
	3: ["#1b9e77","#d95f02","#7570b3"],
	4: ["#1b9e77","#d95f02","#7570b3","#e7298a"],
	5: ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e"],
	6: ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02"],
	7: ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d"],
	8: ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666"]
    },Paired: {
	3: ["#a6cee3","#1f78b4","#b2df8a"],
	4: ["#a6cee3","#1f78b4","#b2df8a","#33a02c"],
	5: ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99"],
	6: ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c"],
	7: ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f"],
	8: ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00"],
	9: ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6"],
	10: ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a"],
	11: ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#ffff99"],
	12: ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#ffff99","#b15928"]
    },Pastel1: {
	3: ["#fbb4ae","#b3cde3","#ccebc5"],
	4: ["#fbb4ae","#b3cde3","#ccebc5","#decbe4"],
	5: ["#fbb4ae","#b3cde3","#ccebc5","#decbe4","#fed9a6"],
	6: ["#fbb4ae","#b3cde3","#ccebc5","#decbe4","#fed9a6","#ffffcc"],
	7: ["#fbb4ae","#b3cde3","#ccebc5","#decbe4","#fed9a6","#ffffcc","#e5d8bd"],
	8: ["#fbb4ae","#b3cde3","#ccebc5","#decbe4","#fed9a6","#ffffcc","#e5d8bd","#fddaec"],
	9: ["#fbb4ae","#b3cde3","#ccebc5","#decbe4","#fed9a6","#ffffcc","#e5d8bd","#fddaec","#f2f2f2"]
    },Pastel2: {
	3: ["#b3e2cd","#fdcdac","#cbd5e8"],
	4: ["#b3e2cd","#fdcdac","#cbd5e8","#f4cae4"],
	5: ["#b3e2cd","#fdcdac","#cbd5e8","#f4cae4","#e6f5c9"],
	6: ["#b3e2cd","#fdcdac","#cbd5e8","#f4cae4","#e6f5c9","#fff2ae"],
	7: ["#b3e2cd","#fdcdac","#cbd5e8","#f4cae4","#e6f5c9","#fff2ae","#f1e2cc"],
	8: ["#b3e2cd","#fdcdac","#cbd5e8","#f4cae4","#e6f5c9","#fff2ae","#f1e2cc","#cccccc"]
    },Set1: {
	3: ["#e41a1c","#377eb8","#4daf4a"],
	4: ["#e41a1c","#377eb8","#4daf4a","#984ea3"],
	5: ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00"],
	6: ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33"],
	7: ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33","#a65628"],
	8: ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33","#a65628","#f781bf"],
	9: ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33","#a65628","#f781bf","#999999"]
    },Set2: {
	3: ["#66c2a5","#fc8d62","#8da0cb"],
	4: ["#66c2a5","#fc8d62","#8da0cb","#e78ac3"],
	5: ["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854"],
	6: ["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f"],
	7: ["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494"],
	8: ["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494","#b3b3b3"]
    },Set3: {
	3: ["#8dd3c7","#ffffb3","#bebada"],
	4: ["#8dd3c7","#ffffb3","#bebada","#fb8072"],
	5: ["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3"],
	6: ["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462"],
	7: ["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69"],
	8: ["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5"],
	9: ["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9"],
	10: ["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd"],
	11: ["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd","#ccebc5"],
	12: ["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd","#ccebc5","#ffed6f"]
    }};

    return colorbrewer;
});

define('charts/surface',[
    "components/legends",
    "utils/utils",
    "utils/datasets",
    "utils/colorbrewer"
],function(Legends, Utils, Datasets, colorbrewer){
    function Surface(data, options){
	this.options = {
	    fill_colors: colorbrewer.Reds[3],
	    has_legend: true
	};

	if(arguments.length > 1){
	    Utils.merge(this.options, options);
	}

	this.dataset = new Datasets.Matrix(data);
	this.ranges = this.dataset.ranges;
    }

    Surface.prototype.generateMesh = function(scales){
	var data = this.dataset.raw;
	var geometry = new THREE.Geometry();
	var color_scale = d3.scale.linear()
	    .domain(this.ranges.z.divide(this.options.fill_colors.length))
	    .range(this.options.fill_colors);
	var colors = [];
	var offset = function(x,y){return x*width+y;};
	var fillFace = function(geometry, p1, p2, p3, colors){
	    var vec0 = new THREE.Vector3(), vec1 = new THREE.Vector3();
	    vec0.subVectors(geometry.vertices[p1],geometry.vertices[p2]);
	    vec1.subVectors(geometry.vertices[p1],geometry.vertices[p3]);
	    vec1.cross(vec0).normalize();
	    var color_arr = [colors[p1], colors[p2], colors[p3]];
	    geometry.faces.push(new THREE.Face3(p1, p2, p3, vec1, color_arr));
	    color_arr = [colors[p3], colors[p2], colors[p1]];
	    geometry.faces.push(new THREE.Face3(p3, p2, p1, vec1.negate(), color_arr));
	}
	var width = data.x.length, height = data.x[0].length;

	for(var i=0;i<width;i++){
	    for(var j=0;j<height;j++){
		geometry.vertices.push(new THREE.Vector3(
		    scales.x(data.x[i][j]),
		    scales.y(data.y[i][j]),
		    scales.z(data.z[i][j])
		));
		colors.push(new THREE.Color(color_scale(data.z[i][j])));
	    }
	}

	for(var x=0;x<width-1;x++){
	    for(var y=0;y<height-1;y++){
		fillFace(geometry, offset(x,y), offset(x+1,y), offset(x,y+1), colors);
		fillFace(geometry, offset(x+1,y), offset(x+1,y+1), offset(x, y+1), colors);
	    }
	}
	var material = new THREE.MeshBasicMaterial({ vertexColors: THREE.VertexColors});
	this.mesh = new THREE.Mesh(geometry, material);
    }

    Surface.prototype.getDataRanges = function(){
	return this.ranges;
    }
    
    Surface.prototype.hasLegend = function(){
	return this.options.has_legend;
    }

    Surface.prototype.getLegend = function(){
	return Legends.generateContinuousLegend(this.ranges.z, this.options.fill_colors);
    }
    
    Surface.prototype.getMesh = function(){
	return this.mesh;
    };

    return Surface;
});

define('charts/wireframe',[
    "components/legends",
    "utils/utils",
    "utils/datasets",
    "utils/colorbrewer"
],function(Legends, Utils, Datasets, colorbrewer){
    function Wireframe(data, options){
	this.options = {
	    name: "wireframe",
	    color: "#999999",
	    thickness: 1,
	    has_legend: true
	};

	if(arguments.length > 1){
	    Utils.merge(this.options, options);
	}

	this.dataset = new Datasets.Matrix(data);
	this.ranges = this.dataset.ranges;
    }

    Wireframe.prototype.generateMesh = function(scales){
	var data = this.dataset.raw;
	var width = data.x.length, height = data.x[0].length;
	var material = new THREE.LineBasicMaterial({ 
	    color: this.options.color,
	    linewidth: this.options.thickness,
	    transparent: true
	});
	var meshes = [];
	for(var i=0;i<width;i++){
	    var geometry = new THREE.Geometry();
	    for(var j=0;j<height;j++){
		geometry.vertices.push(new THREE.Vector3(
		    scales.x(data.x[i][j]),
		    scales.y(data.y[i][j]),
		    scales.z(data.z[i][j])
		));
	    }
	    meshes.push(new THREE.Line(geometry, material));
	}

	for(var j=0;j<height;j++){
	    var geometry = new THREE.Geometry();
	    for(var i=0;i<width;i++){
		geometry.vertices.push(new THREE.Vector3(
		    scales.x(data.x[i][j]),
		    scales.y(data.y[i][j]),
		    scales.z(data.z[i][j])
		));
	    }
	    meshes.push(new THREE.Line(geometry, material));
	}

	this.mesh = meshes;
    };

    Wireframe.prototype.getDataRanges = function(){
	return this.ranges;
    };
    
    Wireframe.prototype.hasLegend = function(){
	return this.options.has_legend;
    };

    Wireframe.prototype.disappear = function(){
	for(var i=0;i<this.mesh.length;i++){
	    this.mesh[i].material.opacity = 0;
	    this.mesh[i].material.needsUpdate = true;
	}
    };

    Wireframe.prototype.appear = function(){
	for(var i=0;i<this.mesh.length;i++){
	    this.mesh[i].material.opacity = 1;
	    this.mesh[i].material.needsUpdate = true;
	}
    };

    Wireframe.prototype.getLegend = function(){
	return Legends.generateDiscreteLegend(this.options.name, this.options.color, this);
    };
    
    Wireframe.prototype.getMesh = function(){
	return this.mesh;
    };

    return Wireframe;
});

//     Underscore.js 1.7.0
//     http://underscorejs.org
//     (c) 2009-2014 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
//     Underscore may be freely distributed under the MIT license.
(function(){var n=this,t=n._,r=Array.prototype,e=Object.prototype,u=Function.prototype,i=r.push,a=r.slice,o=r.concat,l=e.toString,c=e.hasOwnProperty,f=Array.isArray,s=Object.keys,p=u.bind,h=function(n){return n instanceof h?n:this instanceof h?void(this._wrapped=n):new h(n)};"undefined"!=typeof exports?("undefined"!=typeof module&&module.exports&&(exports=module.exports=h),exports._=h):n._=h,h.VERSION="1.7.0";var g=function(n,t,r){if(t===void 0)return n;switch(null==r?3:r){case 1:return function(r){return n.call(t,r)};case 2:return function(r,e){return n.call(t,r,e)};case 3:return function(r,e,u){return n.call(t,r,e,u)};case 4:return function(r,e,u,i){return n.call(t,r,e,u,i)}}return function(){return n.apply(t,arguments)}};h.iteratee=function(n,t,r){return null==n?h.identity:h.isFunction(n)?g(n,t,r):h.isObject(n)?h.matches(n):h.property(n)},h.each=h.forEach=function(n,t,r){if(null==n)return n;t=g(t,r);var e,u=n.length;if(u===+u)for(e=0;u>e;e++)t(n[e],e,n);else{var i=h.keys(n);for(e=0,u=i.length;u>e;e++)t(n[i[e]],i[e],n)}return n},h.map=h.collect=function(n,t,r){if(null==n)return[];t=h.iteratee(t,r);for(var e,u=n.length!==+n.length&&h.keys(n),i=(u||n).length,a=Array(i),o=0;i>o;o++)e=u?u[o]:o,a[o]=t(n[e],e,n);return a};var v="Reduce of empty array with no initial value";h.reduce=h.foldl=h.inject=function(n,t,r,e){null==n&&(n=[]),t=g(t,e,4);var u,i=n.length!==+n.length&&h.keys(n),a=(i||n).length,o=0;if(arguments.length<3){if(!a)throw new TypeError(v);r=n[i?i[o++]:o++]}for(;a>o;o++)u=i?i[o]:o,r=t(r,n[u],u,n);return r},h.reduceRight=h.foldr=function(n,t,r,e){null==n&&(n=[]),t=g(t,e,4);var u,i=n.length!==+n.length&&h.keys(n),a=(i||n).length;if(arguments.length<3){if(!a)throw new TypeError(v);r=n[i?i[--a]:--a]}for(;a--;)u=i?i[a]:a,r=t(r,n[u],u,n);return r},h.find=h.detect=function(n,t,r){var e;return t=h.iteratee(t,r),h.some(n,function(n,r,u){return t(n,r,u)?(e=n,!0):void 0}),e},h.filter=h.select=function(n,t,r){var e=[];return null==n?e:(t=h.iteratee(t,r),h.each(n,function(n,r,u){t(n,r,u)&&e.push(n)}),e)},h.reject=function(n,t,r){return h.filter(n,h.negate(h.iteratee(t)),r)},h.every=h.all=function(n,t,r){if(null==n)return!0;t=h.iteratee(t,r);var e,u,i=n.length!==+n.length&&h.keys(n),a=(i||n).length;for(e=0;a>e;e++)if(u=i?i[e]:e,!t(n[u],u,n))return!1;return!0},h.some=h.any=function(n,t,r){if(null==n)return!1;t=h.iteratee(t,r);var e,u,i=n.length!==+n.length&&h.keys(n),a=(i||n).length;for(e=0;a>e;e++)if(u=i?i[e]:e,t(n[u],u,n))return!0;return!1},h.contains=h.include=function(n,t){return null==n?!1:(n.length!==+n.length&&(n=h.values(n)),h.indexOf(n,t)>=0)},h.invoke=function(n,t){var r=a.call(arguments,2),e=h.isFunction(t);return h.map(n,function(n){return(e?t:n[t]).apply(n,r)})},h.pluck=function(n,t){return h.map(n,h.property(t))},h.where=function(n,t){return h.filter(n,h.matches(t))},h.findWhere=function(n,t){return h.find(n,h.matches(t))},h.max=function(n,t,r){var e,u,i=-1/0,a=-1/0;if(null==t&&null!=n){n=n.length===+n.length?n:h.values(n);for(var o=0,l=n.length;l>o;o++)e=n[o],e>i&&(i=e)}else t=h.iteratee(t,r),h.each(n,function(n,r,e){u=t(n,r,e),(u>a||u===-1/0&&i===-1/0)&&(i=n,a=u)});return i},h.min=function(n,t,r){var e,u,i=1/0,a=1/0;if(null==t&&null!=n){n=n.length===+n.length?n:h.values(n);for(var o=0,l=n.length;l>o;o++)e=n[o],i>e&&(i=e)}else t=h.iteratee(t,r),h.each(n,function(n,r,e){u=t(n,r,e),(a>u||1/0===u&&1/0===i)&&(i=n,a=u)});return i},h.shuffle=function(n){for(var t,r=n&&n.length===+n.length?n:h.values(n),e=r.length,u=Array(e),i=0;e>i;i++)t=h.random(0,i),t!==i&&(u[i]=u[t]),u[t]=r[i];return u},h.sample=function(n,t,r){return null==t||r?(n.length!==+n.length&&(n=h.values(n)),n[h.random(n.length-1)]):h.shuffle(n).slice(0,Math.max(0,t))},h.sortBy=function(n,t,r){return t=h.iteratee(t,r),h.pluck(h.map(n,function(n,r,e){return{value:n,index:r,criteria:t(n,r,e)}}).sort(function(n,t){var r=n.criteria,e=t.criteria;if(r!==e){if(r>e||r===void 0)return 1;if(e>r||e===void 0)return-1}return n.index-t.index}),"value")};var m=function(n){return function(t,r,e){var u={};return r=h.iteratee(r,e),h.each(t,function(e,i){var a=r(e,i,t);n(u,e,a)}),u}};h.groupBy=m(function(n,t,r){h.has(n,r)?n[r].push(t):n[r]=[t]}),h.indexBy=m(function(n,t,r){n[r]=t}),h.countBy=m(function(n,t,r){h.has(n,r)?n[r]++:n[r]=1}),h.sortedIndex=function(n,t,r,e){r=h.iteratee(r,e,1);for(var u=r(t),i=0,a=n.length;a>i;){var o=i+a>>>1;r(n[o])<u?i=o+1:a=o}return i},h.toArray=function(n){return n?h.isArray(n)?a.call(n):n.length===+n.length?h.map(n,h.identity):h.values(n):[]},h.size=function(n){return null==n?0:n.length===+n.length?n.length:h.keys(n).length},h.partition=function(n,t,r){t=h.iteratee(t,r);var e=[],u=[];return h.each(n,function(n,r,i){(t(n,r,i)?e:u).push(n)}),[e,u]},h.first=h.head=h.take=function(n,t,r){return null==n?void 0:null==t||r?n[0]:0>t?[]:a.call(n,0,t)},h.initial=function(n,t,r){return a.call(n,0,Math.max(0,n.length-(null==t||r?1:t)))},h.last=function(n,t,r){return null==n?void 0:null==t||r?n[n.length-1]:a.call(n,Math.max(n.length-t,0))},h.rest=h.tail=h.drop=function(n,t,r){return a.call(n,null==t||r?1:t)},h.compact=function(n){return h.filter(n,h.identity)};var y=function(n,t,r,e){if(t&&h.every(n,h.isArray))return o.apply(e,n);for(var u=0,a=n.length;a>u;u++){var l=n[u];h.isArray(l)||h.isArguments(l)?t?i.apply(e,l):y(l,t,r,e):r||e.push(l)}return e};h.flatten=function(n,t){return y(n,t,!1,[])},h.without=function(n){return h.difference(n,a.call(arguments,1))},h.uniq=h.unique=function(n,t,r,e){if(null==n)return[];h.isBoolean(t)||(e=r,r=t,t=!1),null!=r&&(r=h.iteratee(r,e));for(var u=[],i=[],a=0,o=n.length;o>a;a++){var l=n[a];if(t)a&&i===l||u.push(l),i=l;else if(r){var c=r(l,a,n);h.indexOf(i,c)<0&&(i.push(c),u.push(l))}else h.indexOf(u,l)<0&&u.push(l)}return u},h.union=function(){return h.uniq(y(arguments,!0,!0,[]))},h.intersection=function(n){if(null==n)return[];for(var t=[],r=arguments.length,e=0,u=n.length;u>e;e++){var i=n[e];if(!h.contains(t,i)){for(var a=1;r>a&&h.contains(arguments[a],i);a++);a===r&&t.push(i)}}return t},h.difference=function(n){var t=y(a.call(arguments,1),!0,!0,[]);return h.filter(n,function(n){return!h.contains(t,n)})},h.zip=function(n){if(null==n)return[];for(var t=h.max(arguments,"length").length,r=Array(t),e=0;t>e;e++)r[e]=h.pluck(arguments,e);return r},h.object=function(n,t){if(null==n)return{};for(var r={},e=0,u=n.length;u>e;e++)t?r[n[e]]=t[e]:r[n[e][0]]=n[e][1];return r},h.indexOf=function(n,t,r){if(null==n)return-1;var e=0,u=n.length;if(r){if("number"!=typeof r)return e=h.sortedIndex(n,t),n[e]===t?e:-1;e=0>r?Math.max(0,u+r):r}for(;u>e;e++)if(n[e]===t)return e;return-1},h.lastIndexOf=function(n,t,r){if(null==n)return-1;var e=n.length;for("number"==typeof r&&(e=0>r?e+r+1:Math.min(e,r+1));--e>=0;)if(n[e]===t)return e;return-1},h.range=function(n,t,r){arguments.length<=1&&(t=n||0,n=0),r=r||1;for(var e=Math.max(Math.ceil((t-n)/r),0),u=Array(e),i=0;e>i;i++,n+=r)u[i]=n;return u};var d=function(){};h.bind=function(n,t){var r,e;if(p&&n.bind===p)return p.apply(n,a.call(arguments,1));if(!h.isFunction(n))throw new TypeError("Bind must be called on a function");return r=a.call(arguments,2),e=function(){if(!(this instanceof e))return n.apply(t,r.concat(a.call(arguments)));d.prototype=n.prototype;var u=new d;d.prototype=null;var i=n.apply(u,r.concat(a.call(arguments)));return h.isObject(i)?i:u}},h.partial=function(n){var t=a.call(arguments,1);return function(){for(var r=0,e=t.slice(),u=0,i=e.length;i>u;u++)e[u]===h&&(e[u]=arguments[r++]);for(;r<arguments.length;)e.push(arguments[r++]);return n.apply(this,e)}},h.bindAll=function(n){var t,r,e=arguments.length;if(1>=e)throw new Error("bindAll must be passed function names");for(t=1;e>t;t++)r=arguments[t],n[r]=h.bind(n[r],n);return n},h.memoize=function(n,t){var r=function(e){var u=r.cache,i=t?t.apply(this,arguments):e;return h.has(u,i)||(u[i]=n.apply(this,arguments)),u[i]};return r.cache={},r},h.delay=function(n,t){var r=a.call(arguments,2);return setTimeout(function(){return n.apply(null,r)},t)},h.defer=function(n){return h.delay.apply(h,[n,1].concat(a.call(arguments,1)))},h.throttle=function(n,t,r){var e,u,i,a=null,o=0;r||(r={});var l=function(){o=r.leading===!1?0:h.now(),a=null,i=n.apply(e,u),a||(e=u=null)};return function(){var c=h.now();o||r.leading!==!1||(o=c);var f=t-(c-o);return e=this,u=arguments,0>=f||f>t?(clearTimeout(a),a=null,o=c,i=n.apply(e,u),a||(e=u=null)):a||r.trailing===!1||(a=setTimeout(l,f)),i}},h.debounce=function(n,t,r){var e,u,i,a,o,l=function(){var c=h.now()-a;t>c&&c>0?e=setTimeout(l,t-c):(e=null,r||(o=n.apply(i,u),e||(i=u=null)))};return function(){i=this,u=arguments,a=h.now();var c=r&&!e;return e||(e=setTimeout(l,t)),c&&(o=n.apply(i,u),i=u=null),o}},h.wrap=function(n,t){return h.partial(t,n)},h.negate=function(n){return function(){return!n.apply(this,arguments)}},h.compose=function(){var n=arguments,t=n.length-1;return function(){for(var r=t,e=n[t].apply(this,arguments);r--;)e=n[r].call(this,e);return e}},h.after=function(n,t){return function(){return--n<1?t.apply(this,arguments):void 0}},h.before=function(n,t){var r;return function(){return--n>0?r=t.apply(this,arguments):t=null,r}},h.once=h.partial(h.before,2),h.keys=function(n){if(!h.isObject(n))return[];if(s)return s(n);var t=[];for(var r in n)h.has(n,r)&&t.push(r);return t},h.values=function(n){for(var t=h.keys(n),r=t.length,e=Array(r),u=0;r>u;u++)e[u]=n[t[u]];return e},h.pairs=function(n){for(var t=h.keys(n),r=t.length,e=Array(r),u=0;r>u;u++)e[u]=[t[u],n[t[u]]];return e},h.invert=function(n){for(var t={},r=h.keys(n),e=0,u=r.length;u>e;e++)t[n[r[e]]]=r[e];return t},h.functions=h.methods=function(n){var t=[];for(var r in n)h.isFunction(n[r])&&t.push(r);return t.sort()},h.extend=function(n){if(!h.isObject(n))return n;for(var t,r,e=1,u=arguments.length;u>e;e++){t=arguments[e];for(r in t)c.call(t,r)&&(n[r]=t[r])}return n},h.pick=function(n,t,r){var e,u={};if(null==n)return u;if(h.isFunction(t)){t=g(t,r);for(e in n){var i=n[e];t(i,e,n)&&(u[e]=i)}}else{var l=o.apply([],a.call(arguments,1));n=new Object(n);for(var c=0,f=l.length;f>c;c++)e=l[c],e in n&&(u[e]=n[e])}return u},h.omit=function(n,t,r){if(h.isFunction(t))t=h.negate(t);else{var e=h.map(o.apply([],a.call(arguments,1)),String);t=function(n,t){return!h.contains(e,t)}}return h.pick(n,t,r)},h.defaults=function(n){if(!h.isObject(n))return n;for(var t=1,r=arguments.length;r>t;t++){var e=arguments[t];for(var u in e)n[u]===void 0&&(n[u]=e[u])}return n},h.clone=function(n){return h.isObject(n)?h.isArray(n)?n.slice():h.extend({},n):n},h.tap=function(n,t){return t(n),n};var b=function(n,t,r,e){if(n===t)return 0!==n||1/n===1/t;if(null==n||null==t)return n===t;n instanceof h&&(n=n._wrapped),t instanceof h&&(t=t._wrapped);var u=l.call(n);if(u!==l.call(t))return!1;switch(u){case"[object RegExp]":case"[object String]":return""+n==""+t;case"[object Number]":return+n!==+n?+t!==+t:0===+n?1/+n===1/t:+n===+t;case"[object Date]":case"[object Boolean]":return+n===+t}if("object"!=typeof n||"object"!=typeof t)return!1;for(var i=r.length;i--;)if(r[i]===n)return e[i]===t;var a=n.constructor,o=t.constructor;if(a!==o&&"constructor"in n&&"constructor"in t&&!(h.isFunction(a)&&a instanceof a&&h.isFunction(o)&&o instanceof o))return!1;r.push(n),e.push(t);var c,f;if("[object Array]"===u){if(c=n.length,f=c===t.length)for(;c--&&(f=b(n[c],t[c],r,e)););}else{var s,p=h.keys(n);if(c=p.length,f=h.keys(t).length===c)for(;c--&&(s=p[c],f=h.has(t,s)&&b(n[s],t[s],r,e)););}return r.pop(),e.pop(),f};h.isEqual=function(n,t){return b(n,t,[],[])},h.isEmpty=function(n){if(null==n)return!0;if(h.isArray(n)||h.isString(n)||h.isArguments(n))return 0===n.length;for(var t in n)if(h.has(n,t))return!1;return!0},h.isElement=function(n){return!(!n||1!==n.nodeType)},h.isArray=f||function(n){return"[object Array]"===l.call(n)},h.isObject=function(n){var t=typeof n;return"function"===t||"object"===t&&!!n},h.each(["Arguments","Function","String","Number","Date","RegExp"],function(n){h["is"+n]=function(t){return l.call(t)==="[object "+n+"]"}}),h.isArguments(arguments)||(h.isArguments=function(n){return h.has(n,"callee")}),"function"!=typeof/./&&(h.isFunction=function(n){return"function"==typeof n||!1}),h.isFinite=function(n){return isFinite(n)&&!isNaN(parseFloat(n))},h.isNaN=function(n){return h.isNumber(n)&&n!==+n},h.isBoolean=function(n){return n===!0||n===!1||"[object Boolean]"===l.call(n)},h.isNull=function(n){return null===n},h.isUndefined=function(n){return n===void 0},h.has=function(n,t){return null!=n&&c.call(n,t)},h.noConflict=function(){return n._=t,this},h.identity=function(n){return n},h.constant=function(n){return function(){return n}},h.noop=function(){},h.property=function(n){return function(t){return t[n]}},h.matches=function(n){var t=h.pairs(n),r=t.length;return function(n){if(null==n)return!r;n=new Object(n);for(var e=0;r>e;e++){var u=t[e],i=u[0];if(u[1]!==n[i]||!(i in n))return!1}return!0}},h.times=function(n,t,r){var e=Array(Math.max(0,n));t=g(t,r,1);for(var u=0;n>u;u++)e[u]=t(u);return e},h.random=function(n,t){return null==t&&(t=n,n=0),n+Math.floor(Math.random()*(t-n+1))},h.now=Date.now||function(){return(new Date).getTime()};var _={"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#x27;","`":"&#x60;"},w=h.invert(_),j=function(n){var t=function(t){return n[t]},r="(?:"+h.keys(n).join("|")+")",e=RegExp(r),u=RegExp(r,"g");return function(n){return n=null==n?"":""+n,e.test(n)?n.replace(u,t):n}};h.escape=j(_),h.unescape=j(w),h.result=function(n,t){if(null==n)return void 0;var r=n[t];return h.isFunction(r)?n[t]():r};var x=0;h.uniqueId=function(n){var t=++x+"";return n?n+t:t},h.templateSettings={evaluate:/<%([\s\S]+?)%>/g,interpolate:/<%=([\s\S]+?)%>/g,escape:/<%-([\s\S]+?)%>/g};var A=/(.)^/,k={"'":"'","\\":"\\","\r":"r","\n":"n","\u2028":"u2028","\u2029":"u2029"},O=/\\|'|\r|\n|\u2028|\u2029/g,F=function(n){return"\\"+k[n]};h.template=function(n,t,r){!t&&r&&(t=r),t=h.defaults({},t,h.templateSettings);var e=RegExp([(t.escape||A).source,(t.interpolate||A).source,(t.evaluate||A).source].join("|")+"|$","g"),u=0,i="__p+='";n.replace(e,function(t,r,e,a,o){return i+=n.slice(u,o).replace(O,F),u=o+t.length,r?i+="'+\n((__t=("+r+"))==null?'':_.escape(__t))+\n'":e?i+="'+\n((__t=("+e+"))==null?'':__t)+\n'":a&&(i+="';\n"+a+"\n__p+='"),t}),i+="';\n",t.variable||(i="with(obj||{}){\n"+i+"}\n"),i="var __t,__p='',__j=Array.prototype.join,"+"print=function(){__p+=__j.call(arguments,'');};\n"+i+"return __p;\n";try{var a=new Function(t.variable||"obj","_",i)}catch(o){throw o.source=i,o}var l=function(n){return a.call(this,n,h)},c=t.variable||"obj";return l.source="function("+c+"){\n"+i+"}",l},h.chain=function(n){var t=h(n);return t._chain=!0,t};var E=function(n){return this._chain?h(n).chain():n};h.mixin=function(n){h.each(h.functions(n),function(t){var r=h[t]=n[t];h.prototype[t]=function(){var n=[this._wrapped];return i.apply(n,arguments),E.call(this,r.apply(h,n))}})},h.mixin(h),h.each(["pop","push","reverse","shift","sort","splice","unshift"],function(n){var t=r[n];h.prototype[n]=function(){var r=this._wrapped;return t.apply(r,arguments),"shift"!==n&&"splice"!==n||0!==r.length||delete r[0],E.call(this,r)}}),h.each(["concat","join","slice"],function(n){var t=r[n];h.prototype[n]=function(){return E.call(this,t.apply(this._wrapped,arguments))}}),h.prototype.value=function(){return this._wrapped},"function"==typeof define&&define.amd&&define("underscore",[],function(){return h})}).call(this);
//# sourceMappingURL=underscore-min.map;
define('charts/particles',[
    "underscore",
    "components/legends",
    "utils/utils",
    "utils/datasets",
    "utils/colorbrewer"
],function(_, Legends, Utils, Datasets, colorbrewer){
    function Particles(data, options){
	    this.options = {
	        name: "Particle",
	        color: colorbrewer.Reds[3],
	        size: 0.3,
	        has_legend: true,
		fill_by: null,
		fill_by_range: null
	    };

	    if(arguments.length > 1){
	        Utils.merge(this.options, options);
	    }

	    this.data = data;
    }

    Particles.prototype.generateMesh = function(scales){
	var data = new Datasets.Array(this.data).raw;
	var meshes = [];
	for(var i=0;i<data.x.length;i++){
	    var mesh = new THREE.Mesh(new THREE.SphereGeometry(this.options.size));
	    mesh.position.set(
		scales.x(data.x[i]),
		scales.y(data.y[i]),
		scales.z(data.z[i])
	    );
	    meshes.push(mesh);
	}

	if(this.options.fill_by){
	    var fill_by_column = data[this.options.fill_by];
	    var range;
	    if(_.isNull(this.options.fill_by_range))
		range = _.map(['min', 'max'], function(name){return _[name](fill_by_column);});
	    else
		range = this.options.fill_by_range;

	    var domain = _.range(range[0], range[1]+1, (range[1]-range[0])/(this.options.color.length));
	    var color_scale = d3.scale.linear()
		    .domain(domain)
		    .range(this.options.color);
	    this.mesh = _.map(meshes, _.bind(function(mesh, i){
		var color = color_scale(fill_by_column[i]);
		mesh.material = new THREE.MeshBasicMaterial({transparent:true, color: color});
		return mesh;
	    }, this));
	}else{
	    var color = (_.isArray(this.options.color) ? this.options.color.shift() : this.options.color);
	    var material = new THREE.MeshBasicMaterial({transparent:true, color: color});
	    _.each(meshes, function(mesh){mesh.material = material;});
	    this.mesh = meshes;
	}
    };

    Particles.prototype.getDataRanges = function(){
	    var dataset = new Datasets.Array(this.data);
	    return dataset.ranges;
    };
    
    Particles.prototype.hasLegend = function(){
	    return this.options.has_legend;
    };

    Particles.prototype.disappear = function(){
	_.each(this.mesh, function(mesh){
	    mesh.material.opacity = 0;
	    mesh.material.needsUpdate = true;
	});
    };

    Particles.prototype.appear = function(){
	_.each(this.mesh, function(mesh){
	    mesh.material.opacity = 1;
	});
    };

    Particles.prototype.getLegend = function(){
	    return Legends.generateDiscreteLegend(this.options.name, this.options.color, this);
    };
    
    Particles.prototype.getMesh = function(){
	    return this.mesh;
    };

    return Particles;
});

define('charts/line',[
    "components/legends",
    "utils/utils",
    "utils/range",
    "utils/datasets",
    "utils/colorbrewer"
],function(Legends, Utils, Range, Datasets, colorbrewer){
    function Line(data, options){
	    this.options = {
	        name: "Line",
	        colors: colorbrewer.Blues[3],
	        thickness: 1,
	        has_legend: true
	    };

	    if(arguments.length > 1){
	        Utils.merge(this.options, options);
	    }

	    this.data = data;
	    this.dataset = new Datasets.Array(data);
	    this.ranges = this.dataset.ranges;
    }

    Line.prototype.generateMesh = function(scales){
        var data = new Datasets.Array(this.data).raw;
	    var geometry = new THREE.Geometry();
	    var range = new Range(data.x.length, 0);
	    var color_scale = d3.scale.linear()
	            .domain(range.divide(this.options.colors.length))
	            .range(this.options.colors);
	    for(var i=0;i<data.x.length;i++){
	        geometry.vertices.push(new THREE.Vector3(
		        scales.x(data.x[i]),
		        scales.y(data.y[i]),
		        scales.z(data.z[i])
	        ));
	        geometry.colors.push(new THREE.Color(color_scale(i)));
	    }
	    geometry.colorsNeedUpdate = true;
	    var material = new THREE.LineBasicMaterial({vertexColors:THREE.VertexColors, linewidth:this.options.thickness, transparent:true});
	    this.mesh = new THREE.Line(geometry, material);
    };

    Line.prototype.getDataRanges = function(){
	    return this.ranges;
    };
    
    Line.prototype.hasLegend = function(){
	    return this.options.has_legend;
    };

    Line.prototype.disappear = function(){
	    this.mesh.material.opacity = 0;
	    this.mesh.material.needsUpdate = true;
    };

    Line.prototype.appear = function(){
	    this.mesh.material.opacity = 1;
    };

    Line.prototype.getLegend = function(){
	    return Legends.generateDiscreteLegend(this.options.name, this.options.colors[0], this);
    };
    
    Line.prototype.getMesh = function(){
	    return this.mesh;
    };

    return Line;
});

define('charts/scatter',[
    "components/legends",
    "utils/utils",
    "utils/datasets",
    "utils/colorbrewer"
],function(Legends, Utils, Datasets, colorbrewer){
    function Scatter(data, options){
	this.options = {
	    name: "Scatter",
	    shape: "circle",
	    size: 1.5,
	    stroke_width: 3,
	    stroke_color: "#000000",
	    fill_color: colorbrewer.Reds[3][1],
	    has_legend: true
	};

	if(arguments.length > 1){
	    Utils.merge(this.options, options);
	}

	this.dataset = new Datasets.Array(data);
	this.ranges = this.dataset.ranges;
    }

    Scatter.prototype.generateMesh = function(scales){
	var shape_funcs = {
	    circle: function(ctx){
		ctx.arc(50, 50, 40, 0, Math.PI*2, false);
	    },
	    rect: function(ctx){
		ctx.beginPath();
		ctx.moveTo(20,20);
		ctx.lineTo(80,20);
		ctx.lineTo(80,80);
		ctx.lineTo(20,80);
		ctx.lineTo(20,20);
	    },
	    cross: function(ctx){
		var vertexes = [[35,5],[65,5],[65,35],[95,35],[95,65],[65,65],[65,95],[35,95],[35,65],[5,65],[5,35],[35,35]];
		ctx.moveTo(vertexes[11][0],vertexes[11][1]);
		for(var i=0;i<vertexes.length;i++){
		    ctx.lineTo(vertexes[i][0],vertexes[i][1]);
		}
	    },
	    diamond: function(ctx){
		ctx.moveTo(50,5);
		ctx.lineTo(85,50);
		ctx.lineTo(50,95);
		ctx.lineTo(15,50);
		ctx.lineTo(50,5);
	    }
	};

	var canvas = document.createElement('canvas');
	canvas.width = 100;
	canvas.height = 100;
	var ctx = canvas.getContext('2d');
	ctx.fillStyle = this.options.fill_color;
	shape_funcs[this.options.shape](ctx);
	ctx.fill();
	ctx.lineWidth = this.options.stroke_width;
	ctx.strokeStyle = this.options.stroke_color;
	ctx.stroke();

	var texture = new THREE.Texture(canvas);
	texture.flipY = false;
	texture.needsUpdate = true;
	var material = new THREE.SpriteMaterial({
	    map: texture,
	    size: 10,
	    transparent: true
	});

	var data = this.dataset.raw;
	var meshes = [];
	for(var i=0;i<data.x.length;i++){
	    var sprite = new THREE.Sprite(material);
	    sprite.position.set(
		scales.x(data.x[i]),
		scales.y(data.y[i]),
		scales.z(data.z[i])
	    );
	    var size = this.options.size;
	    sprite.scale.set(size,size,size);
	    meshes.push(sprite);
	}
	this.mesh = meshes;
    };

    Scatter.prototype.getDataRanges = function(){
	return this.ranges;
    };
    
    Scatter.prototype.hasLegend = function(){
	return this.options.has_legend;
    };

    Scatter.prototype.disappear = function(){
	for(var i=0;i<this.mesh.length;i++){
	    this.mesh[i].material.opacity = 0;
	}
    };

    Scatter.prototype.appear = function(){
	for(var i=0;i<this.mesh.length;i++){
	    this.mesh[i].material.opacity = 1;
	}
    };

    Scatter.prototype.getLegend = function(){
	return Legends.generateDiscreteLegend(this.options.name, this.options.fill_color, this);
    };

    Scatter.prototype.getMesh = function(){
	return this.mesh;
    };

    return Scatter;
});

define('shaders/fs',[],function(){return "#ifdef GL_ES__terminate__\
precision highp float;__terminate__\
#endif__terminate__\
__terminate__\
//---------------------------------------------------------__terminate__\
// MACROS__terminate__\
//---------------------------------------------------------__terminate__\
__terminate__\
#define EPS       0.0001__terminate__\
#define PI        3.14159265__terminate__\
#define HALFPI    1.57079633__terminate__\
#define ROOTTHREE 1.73205081__terminate__\
__terminate__\
#define EQUALS(A,B) ( abs((A)-(B)) < EPS )__terminate__\
#define EQUALSZERO(A) ( ((A)<EPS) && ((A)>-EPS) )__terminate__\
__terminate__\
__terminate__\
//---------------------------------------------------------__terminate__\
// CONSTANTS__terminate__\
//---------------------------------------------------------__terminate__\
__terminate__\
// 32 48 64 96 128__terminate__\
#define MAX_STEPS 64__terminate__\
__terminate__\
//#define uTMK 20.0__terminate__\
#define TM_MIN 0.05__terminate__\
__terminate__\
__terminate__\
//---------------------------------------------------------__terminate__\
// SHADER VARS__terminate__\
//---------------------------------------------------------__terminate__\
__terminate__\
varying vec2 vUv;__terminate__\
varying vec3 vPos0; // position in world coords__terminate__\
varying vec3 vPos1; // position in object coords__terminate__\
varying vec3 vPos1n; // normalized 0 to 1, for texture lookup__terminate__\
__terminate__\
uniform vec3 uOffset; // TESTDEBUG__terminate__\
__terminate__\
uniform vec3 uCamPos;__terminate__\
__terminate__\
uniform vec3 uColor;      // color of volume__terminate__\
uniform sampler2D uTex;   // 3D(2D) volume texture__terminate__\
uniform vec3 uTexDim;     // dimensions of texture__terminate__\
__terminate__\
uniform float fPerRow;__terminate__\
uniform float fPerColumn;__terminate__\
__terminate__\
uniform float uTMK;__terminate__\
__terminate__\
float gStepSize;__terminate__\
float gStepFactor;__terminate__\
__terminate__\
//---------------------------------------------------------__terminate__\
// PROGRAM__terminate__\
//---------------------------------------------------------__terminate__\
__terminate__\
// TODO: convert world to local volume space__terminate__\
vec3 toLocal(vec3 p) {__terminate__\
    return p + vec3(0.5);__terminate__\
}__terminate__\
__terminate__\
vec4 sampleVolTex(vec3 pos) {__terminate__\
  pos = pos;__terminate__\
  __terminate__\
  // note: z is up in 3D tex coords, pos.z is tex.y, pos.y is zSlice__terminate__\
  float zSlice = (1.0-pos.y)*(uTexDim.z-1.0);   // float value of slice number, slice 0th to 63rd__terminate__\
__terminate__\
  float x0 = mod(floor(zSlice), fPerRow)*uTexDim.x +__terminate__\
      pos.x*(uTexDim.x-1.0) +__terminate__\
      0.5;__terminate__\
__terminate__\
  float y0 = floor(floor(zSlice)/fPerRow)*uTexDim.y +__terminate__\
      pos.z*(uTexDim.y-1.0) +__terminate__\
      0.5;__terminate__\
__terminate__\
  float width = uTexDim.x*fPerRow;__terminate__\
  float height = uTexDim.y*fPerColumn;__terminate__\
__terminate__\
  float uni_x0 = min(x0/width, 1.0);__terminate__\
  float uni_y0 = min(y0/height, 1.0);__terminate__\
  float uni_x1;__terminate__\
  float uni_y1;__terminate__\
__terminate__\
  if(mod(floor(zSlice)+1.0, fPerRow) == 0.0){__terminate__\
      uni_x1 = min((pos.x*(uTexDim.x-1.0) + 0.5)/width, 1.0);__terminate__\
      uni_y1 = min((y0 + uTexDim.y)/height, 1.0);__terminate__\
  }else{__terminate__\
      uni_x1 = min((x0 + uTexDim.x)/width, 1.0);__terminate__\
      uni_y1 = uni_y0;__terminate__\
  }__terminate__\
__terminate__\
  // get (bi)linear interped texture reads at two slices__terminate__\
  vec4 z0 = texture2D(uTex, vec2(uni_x0, uni_y0));__terminate__\
  vec4 z1 = texture2D(uTex, vec2(uni_x1, uni_y1));__terminate__\
  return mix(z0, z1, fract(zSlice));__terminate__\
}__terminate__\
__terminate__\
vec4 raymarchNoLight(vec3 ro, vec3 rd) {__terminate__\
    vec3 step = rd*gStepSize;__terminate__\
    vec3 pos = ro;__terminate__\
  __terminate__\
    vec4 col = vec4(0.0);__terminate__\
  __terminate__\
    for (int i=0; i<MAX_STEPS; ++i) {__terminate__\
      //float dtm = exp( -uTMK*gStepSize*sampleVolTex(pos) );__terminate__\
      //tm *= dtm;__terminate__\
      //col += (1.0-dtm) * uColor * tm;__terminate__\
      col += sampleVolTex(pos);__terminate__\
      pos += step;__terminate__\
    __terminate__\
      if (__terminate__\
	  pos.x > 1.0 || pos.x < 0.0 ||__terminate__\
	  pos.y > 1.0 || pos.y < 0.0 ||__terminate__\
	  pos.z > 1.0 || pos.z < 0.0)__terminate__\
	break;__terminate__\
    }__terminate__\
  __terminate__\
    if(col.r > 1.0)col.r = 1.0;__terminate__\
    if(col.g > 1.0)col.g = 1.0;__terminate__\
    if(col.b > 1.0)col.b = 1.0;__terminate__\
    return vec4(col.rgb, 1.0);__terminate__\
}__terminate__\
__terminate__\
__terminate__\
void main() {__terminate__\
    // in world coords, just for now__terminate__\
    vec3 ro = vPos1n;__terminate__\
    vec3 rd = normalize( ro - toLocal(uCamPos) );__terminate__\
    //vec3 rd = normalize(ro-uCamPos);__terminate__\
  __terminate__\
    // step_size = root_three / max_steps ; to get through diagonal  __terminate__\
    gStepSize = ROOTTHREE / float(MAX_STEPS);__terminate__\
    gStepFactor = 32.0 * gStepSize;__terminate__\
  __terminate__\
    gl_FragColor = raymarchNoLight(ro, rd);__terminate__\
}__terminate__\
".replace(/__terminate__/g, "\n");});

define('shaders/vs',[],function(){return "#ifdef GL_ES__terminate__\
precision highp float;__terminate__\
#endif__terminate__\
__terminate__\
varying vec2 vUv;__terminate__\
varying vec3 vPos0;__terminate__\
varying vec3 vPos1;__terminate__\
varying vec3 vPos1n;__terminate__\
varying mat4 vObjMatInv;__terminate__\
__terminate__\
void main()__terminate__\
{__terminate__\
  vUv = uv;__terminate__\
  __terminate__\
  gl_Position = projectionMatrix *__terminate__\
                modelViewMatrix *__terminate__\
                vec4(position,1.0);__terminate__\
                __terminate__\
  vPos0 = ( modelMatrix * vec4(position, 1.0) ).xyz;__terminate__\
  vPos1 = position;__terminate__\
  vPos1n = position+vec3(0.5);__terminate__\
  __terminate__\
  //vObjMatInv = inverse(modelMatrix);__terminate__\
}__terminate__\
".replace(/__terminate__/g, "\n");});

define('charts/volume',[
    "underscore",
    "components/legends",
    "utils/utils",
    "utils/range",
    "utils/datasets",
    "utils/colorbrewer",
    "shaders/fs",
    "shaders/vs"
],function(_, Legends, Utils, Range, Datasets, colorbrewer, fs, vs){
    var THREE = window.THREE;

    function Volume(data, _options){
	this.options = {
	    name: "Volume",
	    has_legend: true,
	    width: 100,
	    height: 100,
	    depth: 100,
	    f_per_row: 1,
	    f_per_column: 1,
	    filter: THREE.NearestFilter
	};
	if(arguments.length>1)_.extend(this.options, _options);

	this.data = new Datasets.Compressed(data);
	this.ranges = {x: [0,1], y: [0,1], z: [0,1]};
    }

    Volume.prototype.generateMesh = function(scales, stage){
	var uniforms = (_.bind(function(){
            var voltexDim = new THREE.Vector3(this.options.width, this.options.height, this.options.depth);
            var texture = (_.bind(function(){
		var image = document.createElement("img");
		var voltex = new THREE.Texture(image);
		image.onload = function(){
		    console.log("Texture loading finished");
                    voltex.needsUpdate=true;
		};
		image.src = this.data.raw;
		voltex.minFilter = voltex.magFilter = this.options.filter;
		voltex.wrapS = voltex.wrapT = THREE.ClampToEdgeWrapping;
		voltex.flipX = true;
		voltex.flipY = false;
		return voltex;
            }, this))();

	    var camera = stage.world.camera;

            return {
		uCamPos: {type: "v3", value: camera.position},
		uColor: {type: "v3", value: new THREE.Vector3(1.0, 1.0, 1.0)},
		uTex: {type: "t", value: texture},
		uTexDim: {type: "v3", value: voltexDim},
		fPerRow: {type: "f", value: this.options.f_per_row},
		fPerColumn: {type: "f", value: this.options.f_per_column},
		uOffset: {type: "v3", value: new THREE.Vector3()},
		uTMK: {type: "f", value: 16.0}
            };
        }, this))();

	var material = new THREE.ShaderMaterial({
            uniforms: uniforms,
            vertexShader: vs,
            fragmentShader: fs,
            depthWrite: false
        });

	this.mesh =  new THREE.Mesh(
	    new THREE.CubeGeometry(1, 1, 1),
	    material
	);
	this.mesh.scale.set(20, 20, 20);
    };

    Volume.prototype.getDataRanges = function(){
	return this.ranges;
    };
    
    Volume.prototype.hasLegend = function(){
	return this.options.has_legend;
    };

    Volume.prototype.disappear = function(){
	this.mesh.material.opacity = 0;
	this.mesh.material.needsUpdate = true;
    };

    Volume.prototype.appear = function(){
	this.mesh.material.opacity = 1;
    };

    Volume.prototype.getLegend = function(){
	return Legends.generateDiscreteLegend(this.options.name, "#000", this);
    };
    
    Volume.prototype.getMesh = function(){
	return this.mesh;
    };

    return Volume;
});

define('charts/cylinder',[
    "components/legends",
    "utils/utils",
    "utils/datasets",
    "utils/colorbrewer"
],function(Legends, Utils, Datasets, colorbrewer){
    function Cylinder(data, options){
	this.options = {
	    name: "Cylinder",
	    color: "#756bb1",
	    size: 0.3,
	    has_legend: true
	};

	if(arguments.length > 1){
	    Utils.merge(this.options, options);
	}

	this.data = data;
	this.dataset = new Datasets.Array(data);
	this.ranges = this.dataset.ranges;
    }

    Cylinder.prototype.generateMesh = function(scales){
	    var data = new Datasets.Array(this.data).raw;
	    var geometry = new THREE.Geometry();
	    for(var i=0;i<data.x.length;i++){
            var height = Math.abs(scales.x(data.height[i]) - scales.x(0));
            var rad = Math.abs(scales.x(data.radius[i]) - scales.x(0));
	        var mesh = new THREE.Mesh(new THREE.CylinderGeometry(rad,rad,height,16));
	        mesh.position.set(
		        scales.x(data.x[i]),
		        scales.y(data.y[i]),
		        scales.z(data.z[i])
	        );
            mesh.useQuaternion = true;
            var axis = new THREE.Vector3(data.x_rad[i], data.y_rad[i], data.z_rad[i]).normalize();
            var angle = data.angle[i];
            var q = new THREE.Quaternion();
            q.setFromAxisAngle(axis, angle);
            mesh.rotation.setFromQuaternion(q);
	        THREE.GeometryUtils.merge(geometry, mesh);
        }
        var material = new THREE.MeshLambertMaterial({transparent:true, color: this.options.color});
	this.mesh = new THREE.Mesh(geometry, material);
    };

    Cylinder.prototype.getDataRanges = function(){
	return this.ranges;
    };
    
    Cylinder.prototype.hasLegend = function(){
	return this.options.has_legend;
    };

    Cylinder.prototype.disappear = function(){
	this.mesh.material.opacity = 0;
	this.mesh.material.needsUpdate = true;
    };

    Cylinder.prototype.appear = function(){
	this.mesh.material.opacity = 1;
    };

    Cylinder.prototype.getLegend = function(){
	return Legends.generateDiscreteLegend(this.options.name, this.options.color, this);
    };
    
    Cylinder.prototype.getMesh = function(){
	return this.mesh;
    };

    return Cylinder;
});

define('charts/debug_object',[
    "components/legends",
    "utils/utils",
    "utils/datasets",
    "utils/colorbrewer"
],function(Legends, Utils, Datasets, colorbrewer){
    /*
     * Experimental Implementation of Debug Objects whose material is wireframe
     * DO NOT APPLY THIS OBJECT TO PRACTICAL USE
     *
     * column layout:
     *   [{x: 0, y: 0, z: 0, type: "box", options: {width: 1, height: 1}}, {}, {}, ...]
     *   optional: x_rad, y_rad, z_rad are set when option.rotate == true
     */

    function Debug_Object(data, options){
	this.options = {
	    name: "objects",
	    color: "#000",
	    has_legend: true,
	    rotate: false
	};

	if(arguments.length > 1){
	    Utils.merge(this.options, options);
	}

	this.data = data;
	this.dataset = new Datasets.Array(data);
	this.ranges = this.dataset.ranges;
    }

    Debug_Object.prototype.generateMesh = function(scales){
	var data = new Datasets.Array(this.data).raw;
	var geometry = new THREE.Geometry();
	for(var i=0;i<data.x.length;i++){
	    var geo, options;
	    switch(data.type[i]){
	    case "box":
		options = Utils.merge({
		    width: 1,
		    height: 1,
		    depth: 1,
		    widthSegments: 1,
		    heightSegments: 1,
		    depthSegments: 1
		}, data.options[i]);
		geo = new THREE.BoxGeometry(
		    Math.abs(scales.x(options.width) - scales.x(0)),
		    Math.abs(scales.y(options.height) - scales.y(0)),
		    Math.abs(scales.z(options.depth) - scales.z(0)),
		    options.widthSegments,
		    options.heightSegments,
		    options.depthSegments
		);
		break;

	    case "cylinder":
		options = Utils.merge({
		    radius: 1,
		    height: 100,
		    radiusSegments: 8,
		    heightSegments: 1
		}, data.options[i]);
		geo = new THREE.CylinderGeometry(
		    Math.abs(scales.x(options.radius) - scales.x(0)),
		    Math.abs(scales.x(options.radius) - scales.x(0)),
		    Math.abs(scales.y(options.height) - scales.y(0)),
		    options.radiusSegments,
		    options.heightSegments
		);
		break;

	    case "sphere":
		options = Utils.merge({
		    radius: 1,
		    horizontalSegments: 8,
		    verticalSegments: 6
		}, data.options[i]);
		geo = new THREE.SphereGeometry(
		    Math.abs(scales.x(options.radius) - scales.x(0)),
		    options.horizontalSegments,
		    options.verticalSegments
		);
		break;

	    case "plane":
		options = Utils.merge({
		    width: 1,
		    height: 1,
		    widthSegments: 1,
		    heightSegments: 1
		}, data.options[i]);
		geo = new THREE.PlaneGeometry(
		    Math.abs(scales.x(options.width) - scales.x(0)),
		    Math.abs(scales.y(options.height) - scales.y(0)),
		    options.widthSegments,
		    options.heightSegments
		);
		break;
	    }

	    var mesh = new THREE.Mesh(geo);

	    mesh.position = new THREE.Vector3(
		scales.x(data.x[i]),
		scales.y(data.y[i]),
		scales.z(data.z[i])
	    );

	    if(this.options.rotate != false){
		mesh.useQuaternion = true;
		var axis = new THREE.Vector3(data.x_rad[i], data.y_rad[i], data.z_rad[i]).normalize();
		var angle = data.angle[i];
		var q = new THREE.Quaternion();
		q.setFromAxisAngle(axis, angle);
		mesh.rotation.setFromQuaternion(q);
	    }
	    THREE.GeometryUtils.merge(geometry, mesh);
        }

        var material = new THREE.MeshPhongMaterial({
            color: this.options.color,
            wireframe: true
	});
	this.mesh = new THREE.Mesh(geometry, material);
    };

    Debug_Object.prototype.getDataRanges = function(){
	return this.ranges;
    };
    
    Debug_Object.prototype.hasLegend = function(){
	return this.options.has_legend;
    };

    Debug_Object.prototype.disappear = function(){
	this.mesh.material.opacity = 0;
	this.mesh.material.needsUpdate = true;
    };

    Debug_Object.prototype.appear = function(){
	this.mesh.material.opacity = 1;
    };

    Debug_Object.prototype.getLegend = function(){
	return Legends.generateDiscreteLegend(this.options.name, this.options.color, this);
    };
    
    Debug_Object.prototype.getMesh = function(){
	return this.mesh;
    };

    return Debug_Object;
});

define('quick/base',[],function(){
    /********************************
      Base function of all quick functions
     **********************************/
    Base = function(){
	options = {};
	this.options = {};

	//setters
	this.width = function(_){
	    this.options.width = _;
	    options = this.options;
	    return this;
	};

	this.height = function(_){
	    this.options.height = _;
	    options = this.options;
	    return this;
	};

	this.bg_color = function(_){
	    this.options.bg_color = _;
	    options = this.options;
	    return this;
	}

	this.legend = function(_){
	    this.options.legend = _;
	    options = this.options;
	    return this;
	}
    }
    return Base;
});

define('quick/surface_plot',[
    "components/stage",
    "quick/base",
    "charts/surface",
    "utils/utils"
],function(Stage, Base, Surface, Utils){

    function SurfacePlot(selection){
	selection.each(function(data){
	    var stage = new Stage(this);
	    stage.add(new Surface(data, options));
	    stage.render();
	});
    }

    SurfacePlot.fill_colors = function(_){
	this.options.fill_colors = _;
	options = this.options;
	return this;
    }

    SurfacePlot.has_legend = function(_){
	this.options.has_legend = _;
	options = this.options;
	return this;
    }

    Utils.mixin(SurfacePlot, Base);

    return SurfacePlot;
});

define('quick/wireframe_plot',[
    "components/stage",
    "quick/base",
    "charts/wireframe",
    "utils/utils"
],function(Stage, Base, Wireframe, Utils){

    function WireframePlot(selection){
	selection.each(function(data){
	    var stage = new Stage(this);
	    stage.add(new Wireframe(data, options));
	    stage.render();
	});
    }

    WireframePlot.data_name = function(_){
	this.options.name = _;
	options = this.options;
	return this;
    }

    WireframePlot.color = function(_){
	this.options.color = _;
	options = this.options;
	return this;
    }

    WireframePlot.thickness = function(_){
	this.options.thickness = _;
	options = this.options;
	return this;
    }

    WireframePlot.has_legend = function(_){
	this.options.has_legend = _;
	options = this.options;
	return this;
    }

    Utils.mixin(WireframePlot, Base);

    return WireframePlot;
});

define('quick/particles_plot',[
    "components/stage",
    "quick/base",
    "charts/particles",
    "utils/utils"
],function(Stage, Base, Particles, Utils){

    function ParticlesPlot(selection){
	selection.each(function(data){
	    var stage = new Stage(this);
	    stage.add(new Particles(data, options));
	    stage.render();
	});
    }

    ParticlesPlot.color = function(_){
	this.options.color = _;
	options = this.options;
	return this;
    }

    ParticlesPlot.size = function(_){
	this.options.size = _;
	options = this.options;
	return this;
    }

    ParticlesPlot.has_legend = function(_){
	this.options.has_legend = _;
	options = this.options;
	return this;
    }

    Utils.mixin(ParticlesPlot, Base);

    return ParticlesPlot;
});

define('quick/line_plot',[
    "components/stage",
    "quick/base",
    "charts/line",
    "utils/utils"
],function(Stage, Base, Line, Utils){

    function LinePlot(selection){
	selection.each(function(data){
	    var stage = new Stage(this);
	    stage.add(new Line(data, options));
	    stage.render();
	});
    };

    LinePlot.data_name = function(_){
	this.options.name = _;
	var options = this.options;
	return this;
    };

    LinePlot.colors = function(_){
	this.options.colors = _;
	var options = this.options;
	return this;
    };

    LinePlot.thickness = function(_){
	this.options.thickness = _;
	var options = this.options;
	return this;
    };

    LinePlot.has_legend = function(_){
	this.options.has_legend = _;
	var options = this.options;
	return this;
    };

    Utils.mixin(LinePlot, Base);

    return LinePlot;
});

define('quick/scatter_plot',[
    "components/stage",
    "quick/base",
    "charts/scatter",
    "utils/utils"
],function(Stage, Base, Scatter, Utils){

    function ScatterPlot(selection){
	selection.each(function(data){
	    var stage = new Stage(this);
	    stage.add(new Scatter(data, options));
	    stage.render();
	});
    }

    ScatterPlot.data_name = function(_){
	this.options.name = _;
	options = this.options;
	return this;
    }

    ScatterPlot.shape = function(_){
	this.options.shape = _;
	options = this.options;
	return this;
    }

    ScatterPlot.size = function(_){
	this.options.size = _;
	options = this.options;
	return this;
    }

    ScatterPlot.stroke_width = function(_){
	this.options.stroke_width = _;
	options = this.options;
	return this;
    }

    ScatterPlot.stroke_color = function(_){
	this.options.stroke_color = _;
	options = this.options;
	return this;
    }

    ScatterPlot.fill_color = function(_){
	this.options.fill_color = _;
	options = this.options;
	return this;
    }

    ScatterPlot.has_legend = function(_){
	this.options.has_legend = _;
	options = this.options;
	return this;
    }

    Utils.mixin(ScatterPlot, Base);

    return ScatterPlot;
});

define('embed/embed',[
    "components/stage",
    "charts/surface",
    "charts/wireframe",
    "charts/scatter",
    "charts/particles",
    "charts/line",
    "charts/cylinder",
    "charts/debug_object",
    "charts/volume"
],function(Stage, Surface, Wireframe, Scatter, Particles, Line, Cylinder, DebugObject, Volume){
    function Embed(){
	return this;
    }

    Embed.parse = function(element_name, model){
	var selection = d3.select(element_name);
	var stage = new Stage(selection[0][0],model.options);
	var plots = model.plots;
	var plot_types = {
	    Surface: Surface,
	    Wireframe: Wireframe,
	    Scatter: Scatter,
	    Particles: Particles,
	    Line: Line,
	    Cylinder: Cylinder,
	    DebugObject: DebugObject,
	    Volume: Volume
	};
	for(var i=0;i<plots.length;i++){
	    var plot = new (plot_types[plots[i].type])(plots[i].data,plots[i].options);
	    stage.add(plot);
	}
	return stage;
    };

    return Embed;
});

define('embed/nyaplot',[
    "components/stage",
    "charts/surface",
    "charts/wireframe",
    "charts/scatter",
    "charts/particles",
    "charts/line"
],function(Stage, Surface, Wireframe, Scatter, Particles, Line){
	var diagrams = {
	    surface: {chart: Surface, dtype: 'Matrix'},
	    wireframe: {chart: Wireframe, dtype: 'Matrix'},
	    scatter: {chart: Scatter, dtype: 'Array'},
	    particles: {chart: Particles, dtype: 'Array'},
	    line: {chart: Line, dtype: 'Array'}
	};

    function Pane(parent, _options){
        this.uuid = Nyaplot.uuid.v4();
        this.stage = new Stage(parent[0][0], _options);
        this.rendered = false;
    }

    Pane.prototype.addDiagram = function(type, df_id, options){
        var data = ({
            'Matrix': prepareMatrix,
            'Array': prepareArray
        }[diagrams[type].dtype])(df_id, options);

        var plot = new (diagrams[type].chart)(data, options);
        this.stage.add(plot);
    };

    function initComponent(df_id, options){
        var df = Nyaplot.Manager.getData(df_id);
        var uuid = Nyaplot.uuid.v4();
        var x = df.columnWithFilters(uuid, options.x);
        var y = df.columnWithFilters(uuid, options.y);
        var z = df.columnWithFilters(uuid, options.z);
        return {x:x, y:y, z:z};
    }

    function prepareArray(df_id, _options){
        var columns = initComponent(df_id, _options);
        return {x: columns.x, y:columns.y, z:columns.z};
    }

    function prepareMatrix(df_id, _options){
        var mat = {};
        var columns = initComponent(df_id, _options);
        ['x','y'].forEach(function(label){
            var column = columns[label];
            var values = column.filter(function(x,i){return column.indexOf(x)==i;});
            var continuous_num = values.map(function(val){
                var first = column.indexOf(val);
                var seek = first;
                while(column[seek]==val)seek++;
                return seek-first;
            });
            if(continuous_num.every(function(val){return (val==continuous_num[0] && val != 1);})){
                var len = continuous_num[0];
                var generate_mat = function(arr, length){
                    var result = [];
                    for(var i=0;i<arr.length;i+=length){
                        result.push(arr.slice(i, i+length));
                    }
                    return result;
                };
                mat.x = generate_mat(columns['x'], len);
                mat.y = generate_mat(columns['y'], len);
                mat.z = generate_mat(columns['z'], len);
                return;
            }
        });
        return mat;
    }

    Pane.prototype.addFilter = function(target, options){
        return;
    };

    Pane.prototype.update = function(){
        if(!this.rendered){
            this.stage.render();
            this.rendered=true;
        }
        return;
    };

    return {
        pane: Pane
    };
});

define('main',['require','exports','module','components/stage','charts/surface','charts/wireframe','charts/particles','charts/line','charts/scatter','charts/volume','charts/cylinder','charts/debug_object','quick/surface_plot','quick/wireframe_plot','quick/particles_plot','quick/line_plot','quick/scatter_plot','utils/database','embed/embed','embed/nyaplot'],function(require, exports, module){
    var Elegans = {};

    /***************************
      Prototype Objects for plotting
      e.g. var stage = new Elegant.Stage(d3.select("#vis"), {width:500, height:500});
           stage.add(new Elegant.Surface(data, {fill_colors:[#000000, #ffffff]}));
           stage.render();
    ****************/

    Elegans.Stage = require("components/stage");
    Elegans.Surface = require("charts/surface");
    Elegans.Wireframe = require("charts/wireframe");
    Elegans.Particles = require("charts/particles");
    Elegans.Line = require("charts/line");
    Elegans.Scatter = require("charts/scatter");
    Elegans.Volume = require("charts/volume");
    Elegans.Cylinder = require("charts/cylinder");
    Elegans.DebugObject = require("charts/debug_object");

    /***************************
      Functions for quick plotting with method chain style  
      e.g. d3.select('#vis').datum(data).call(Elegans.SurfacePlot);
    ****************/

    Elegans.SurfacePlot = require("quick/surface_plot");
    Elegans.WireframePlot = require("quick/wireframe_plot");
    Elegans.ParticlesPlot = require("quick/particles_plot");
    Elegans.LinePlot = require("quick/line_plot");
    Elegans.ScatterPlot = require("quick/scatter_plot");

    /***************************
       Prototype Object for embedding to other language.
       e.g. var model = [{plots:[{type:"Surface",data={[...]},options={...}}],option:{...}}]
            Elegans.Embed.parse(model).render();
    ****************/

    Elegans.DataBase = require("utils/database");
    Elegans.Embed = require("embed/embed");
    Elegans.Nya = require("embed/nyaplot");

    return Elegans;
});

return require('main');
}));